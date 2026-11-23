// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Pooya Parsa <pooya@pi0.io>
// Shared-memory SPSC sound PCM ring between the kernel worker (SoundRingWriter)
// and the owner or AudioWorklet thread (SoundRingReader).
//
// Layout: SOUND_RING_HEADER_BYTES of Int32 control words followed by a
// power-of-two byte buffer. Every chunk record begins at a 16-byte-aligned
// offset inside the buffer with a 4-word header [seq, rate, channels_format,
// byteLen] followed by `byteLen` payload bytes (wrapped at capacity and padded
// to 16 bytes). Since capacity is a multiple of 16 >= 1024, the 4-word header
// never straddles the end of the buffer.

import { SOUND_FORMAT_U8, type SoundFormat } from "./sound.ts";

export const SOUND_RING_HEAD = 0;
export const SOUND_RING_TAIL = 1;
export const SOUND_RING_QUEUED_FRAMES = 2;
export const SOUND_RING_DROPPED = 3;

export const SOUND_RING_HEADER_BYTES = 64;
export const SOUND_RING_DEFAULT_BYTES = 1 << 18; // 256 KiB (~1.36 s @ 48 kHz S16_LE stereo)

export interface SoundRingRecord {
	seq: number;
	format: SoundFormat;
	data: Uint8Array;
}

/** Allocate a SharedArrayBuffer backing a SoundRing (power-of-two data capacity). */
export function createSoundRing(capacityBytes: number = SOUND_RING_DEFAULT_BYTES): SharedArrayBuffer {
	if (!Number.isInteger(capacityBytes) || capacityBytes < 1024 || (capacityBytes & (capacityBytes - 1)) !== 0) {
		throw new RangeError(`sound ring capacity must be a power of two >= 1024, got ${capacityBytes}`);
	}
	return new SharedArrayBuffer(SOUND_RING_HEADER_BYTES + capacityBytes);
}

function ringViews(buffer: SharedArrayBuffer): { ctrl: Int32Array; hdr: Int32Array; data: Uint8Array; capacity: number; mask: number } {
	const capacity = buffer.byteLength - SOUND_RING_HEADER_BYTES;
	if (capacity < 1024 || (capacity & (capacity - 1)) !== 0) {
		throw new RangeError(`invalid sound ring byteLength ${buffer.byteLength}`);
	}
	return {
		ctrl: new Int32Array(buffer, 0, SOUND_RING_HEADER_BYTES >> 2),
		hdr: new Int32Array(buffer),
		data: new Uint8Array(buffer, SOUND_RING_HEADER_BYTES, capacity),
		capacity,
		mask: capacity - 1,
	};
}

function decodeFormat(rate: number, packed: number): SoundFormat {
	const channels = (packed >> 8) & 0xff;
	const format = packed & 0xff;
	const sampleBytes = format === SOUND_FORMAT_U8 ? 1 : 2;
	return { rate, channels, format, frameBytes: sampleBytes * channels };
}

/** Single-producer writer in the kernel worker thread. */
export class SoundRingWriter {
	readonly buffer: SharedArrayBuffer;
	readonly #ctrl: Int32Array;
	readonly #hdr: Int32Array;
	readonly #data: Uint8Array;
	readonly #capacity: number;
	readonly #mask: number;

	constructor(buffer: SharedArrayBuffer) {
		this.buffer = buffer;
		const v = ringViews(buffer);
		this.#ctrl = v.ctrl;
		this.#hdr = v.hdr;
		this.#data = v.data;
		this.#capacity = v.capacity;
		this.#mask = v.mask;
	}

	/**
	 * Append one PCM chunk record. Returns true if the reader appears asleep on
	 * an empty ring (Dekker check: Atomics.store(HEAD) then Atomics.load(TAIL)),
	 * so the caller should send a coalesced "sound-wake" if draining via callbacks.
	 */
	write(seq: number, format: SoundFormat, bytes: Uint8Array): boolean {
		const n = bytes.byteLength;
		const recBytes = 16 + ((n + 15) & ~15);
		if (n <= 0 || recBytes > this.#capacity) return false;
		const head = Atomics.load(this.#ctrl, SOUND_RING_HEAD);
		const tail0 = Atomics.load(this.#ctrl, SOUND_RING_TAIL);
		const used = (head - tail0) | 0;
		if (this.#capacity - used < recBytes) {
			Atomics.add(this.#ctrl, SOUND_RING_DROPPED, n);
			return false;
		}
		const pos = head & this.#mask;
		const wi = (SOUND_RING_HEADER_BYTES + pos) >> 2;
		this.#hdr[wi] = seq | 0;
		this.#hdr[wi + 1] = format.rate | 0;
		this.#hdr[wi + 2] = ((format.channels & 0xff) << 8) | (format.format & 0xff);
		this.#hdr[wi + 3] = n | 0;
		const dataOff = (pos + 16) & this.#mask;
		const first = Math.min(n, this.#capacity - dataOff);
		this.#data.set(bytes.subarray(0, first), dataOff);
		if (first < n) this.#data.set(bytes.subarray(first), 0);
		const nextHead = (head + recBytes) | 0;
		Atomics.store(this.#ctrl, SOUND_RING_HEAD, nextHead);
		const tail1 = Atomics.load(this.#ctrl, SOUND_RING_TAIL);
		return ((nextHead - tail1) | 0) === recBytes;
	}

	/** Downstream queued frames reported atomically by the consumer. */
	queuedFrames(): number {
		return Atomics.load(this.#ctrl, SOUND_RING_QUEUED_FRAMES);
	}

	/** Bytes dropped because the ring was full. */
	droppedBytes(): number {
		return Atomics.load(this.#ctrl, SOUND_RING_DROPPED);
	}
}

/** Single-consumer reader on the owner thread or inside an AudioWorkletProcessor. */
export class SoundRingReader {
	readonly buffer: SharedArrayBuffer;
	readonly #ctrl: Int32Array;
	readonly #hdr: Int32Array;
	readonly #data: Uint8Array;
	readonly #capacity: number;
	readonly #mask: number;

	constructor(buffer: SharedArrayBuffer) {
		this.buffer = buffer;
		const v = ringViews(buffer);
		this.#ctrl = v.ctrl;
		this.#hdr = v.hdr;
		this.#data = v.data;
		this.#capacity = v.capacity;
		this.#mask = v.mask;
	}

	/** Bytes (including 16-byte record headers + padding) currently readable. */
	availableBytes(): number {
		const head = Atomics.load(this.#ctrl, SOUND_RING_HEAD);
		const tail = Atomics.load(this.#ctrl, SOUND_RING_TAIL);
		return (head - tail) | 0;
	}

	/** Update downstream queued frames so the worker reads delay without postMessage. */
	setQueuedFrames(frames: number): void {
		Atomics.store(this.#ctrl, SOUND_RING_QUEUED_FRAMES, Math.max(0, Math.round(frames || 0)));
	}

	/** Bytes dropped by the producer because the ring overflowed. */
	droppedBytes(): number {
		return Atomics.load(this.#ctrl, SOUND_RING_DROPPED);
	}

	/** Read one record (into `scratch` if provided and large enough), or null if empty. */
	readNext(scratch?: Uint8Array): SoundRingRecord | null {
		const head = Atomics.load(this.#ctrl, SOUND_RING_HEAD);
		const tail = Atomics.load(this.#ctrl, SOUND_RING_TAIL);
		if (head === tail) return null;
		const pos = tail & this.#mask;
		const wi = (SOUND_RING_HEADER_BYTES + pos) >> 2;
		const seq = this.#hdr[wi]!;
		const format = decodeFormat(this.#hdr[wi + 1]!, this.#hdr[wi + 2]!);
		const n = this.#hdr[wi + 3]!;
		const out = scratch && scratch.byteLength >= n ? scratch.subarray(0, n) : new Uint8Array(n);
		const dataOff = (pos + 16) & this.#mask;
		const first = Math.min(n, this.#capacity - dataOff);
		out.set(this.#data.subarray(dataOff, dataOff + first), 0);
		if (first < n) out.set(this.#data.subarray(0, n - first), first);
		Atomics.store(this.#ctrl, SOUND_RING_TAIL, (tail + 16 + ((n + 15) & ~15)) | 0);
		return { seq, format, data: out };
	}

	/**
	 * Drain all currently available records in order until the ring is empty
	 * (paired with SoundRingWriter's Dekker check so no tail write is stranded).
	 */
	drain(onRecord: (seq: number, format: SoundFormat, data: Uint8Array) => number | void): void {
		for (;;) {
			const rec = this.readNext();
			if (!rec) return;
			const queued = onRecord(rec.seq, rec.format, rec.data);
			if (typeof queued === "number" && Number.isFinite(queued)) this.setQueuedFrames(queued);
		}
	}
}
