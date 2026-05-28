"""Download specific files from the Cholec80 ZIP archive over HTTP without
fetching the whole 75 GB blob. Uses Range requests against the ZIP's central
directory to locate target entries, then streams just those byte ranges.
"""
from __future__ import annotations

import io
import os
import struct
import sys
import zlib
from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.zip"
OUT_DIR = Path("data/cholec80")
TARGETS = [f"video{i:02d}.mp4" for i in range(61, 71)]  # video61..video70 (test split)

EOCD_SIG = b"PK\x05\x06"
EOCD64_LOCATOR_SIG = b"PK\x06\x07"
EOCD64_SIG = b"PK\x06\x06"
CDH_SIG = b"PK\x01\x02"
LFH_SIG = b"PK\x03\x04"


def http_range(url: str, start: int, end: int) -> bytes:
    req = Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urlopen(req) as r:
        return r.read()


def http_size(url: str) -> int:
    req = Request(url, method="HEAD")
    with urlopen(req) as r:
        return int(r.headers["Content-Length"])


def find_eocd(tail: bytes) -> int:
    idx = tail.rfind(EOCD_SIG)
    if idx < 0:
        raise RuntimeError("EOCD not found in tail")
    return idx


def parse_central_directory(url: str, total_size: int):
    tail_len = min(65 * 1024, total_size)
    tail = http_range(url, total_size - tail_len, total_size - 1)
    eocd_off = find_eocd(tail)
    eocd = tail[eocd_off:eocd_off + 22]
    (sig, disk, cd_disk, n_disk, n_total, cd_size, cd_off, comment_len) = struct.unpack(
        "<IHHHHIIH", eocd
    )

    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or n_total == 0xFFFF:
        loc_pos = tail.rfind(EOCD64_LOCATOR_SIG, 0, eocd_off)
        if loc_pos < 0:
            raise RuntimeError("ZIP64 locator missing")
        (_, _, eocd64_off, _) = struct.unpack("<IIQI", tail[loc_pos:loc_pos + 20])
        eocd64 = http_range(url, eocd64_off, eocd64_off + 56 - 1)
        sig64, size_of_record, ver_made, ver_need, disk64, cd_disk64, n_disk64, n_total64, cd_size64, cd_off64 = struct.unpack(
            "<I Q H H I I Q Q Q Q", eocd64[:56]
        )
        cd_off, cd_size, n_total = cd_off64, cd_size64, n_total64

    print(f"Central directory: offset={cd_off}, size={cd_size}, entries={n_total}", flush=True)

    cd_bytes = http_range(url, cd_off, cd_off + cd_size - 1)
    return cd_bytes, n_total


def iter_central_entries(cd_bytes: bytes):
    pos = 0
    while pos < len(cd_bytes):
        if cd_bytes[pos:pos + 4] != CDH_SIG:
            raise RuntimeError(f"Bad CDH signature at {pos}")
        header = cd_bytes[pos:pos + 46]
        (sig, ver_made, ver_need, flags, method, mtime, mdate, crc32,
         comp_size, uncomp_size, name_len, extra_len, comment_len,
         disk_no, int_attr, ext_attr, lfh_off) = struct.unpack("<IHHHHHHIIIHHHHHII", header)

        name = cd_bytes[pos + 46:pos + 46 + name_len].decode("utf-8", errors="replace")
        extra = cd_bytes[pos + 46 + name_len:pos + 46 + name_len + extra_len]

        # ZIP64 extended info
        if (comp_size == 0xFFFFFFFF or uncomp_size == 0xFFFFFFFF or lfh_off == 0xFFFFFFFF):
            ep = 0
            while ep + 4 <= len(extra):
                tag, sz = struct.unpack("<HH", extra[ep:ep + 4])
                if tag == 0x0001:
                    vals = []
                    o = ep + 4
                    if uncomp_size == 0xFFFFFFFF:
                        uncomp_size = struct.unpack("<Q", extra[o:o + 8])[0]; o += 8
                    if comp_size == 0xFFFFFFFF:
                        comp_size = struct.unpack("<Q", extra[o:o + 8])[0]; o += 8
                    if lfh_off == 0xFFFFFFFF:
                        lfh_off = struct.unpack("<Q", extra[o:o + 8])[0]; o += 8
                    break
                ep += 4 + sz

        yield {
            "name": name,
            "method": method,
            "crc32": crc32,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "lfh_off": lfh_off,
        }
        pos += 46 + name_len + extra_len + comment_len


def download_entry(url: str, entry: dict, out_path: Path):
    lfh_head = http_range(url, entry["lfh_off"], entry["lfh_off"] + 30 - 1)
    if lfh_head[:4] != LFH_SIG:
        raise RuntimeError(f"Bad LFH for {entry['name']}")
    (sig, ver, flags, method, mtime, mdate, crc32, comp_size, uncomp_size,
     name_len, extra_len) = struct.unpack("<IHHHHHIIIHH", lfh_head)

    data_start = entry["lfh_off"] + 30 + name_len + extra_len
    data_end = data_start + entry["comp_size"] - 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")

    print(f"Downloading {entry['name']} ({entry['comp_size'] / 1e9:.2f} GB compressed)...", flush=True)

    req = Request(url, headers={"Range": f"bytes={data_start}-{data_end}"})
    decompress = (entry["method"] == 8)
    dec = zlib.decompressobj(-zlib.MAX_WBITS) if decompress else None
    total = 0
    last_print = 0
    with urlopen(req) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            if dec is not None:
                chunk = dec.decompress(chunk)
            f.write(chunk)
            total += len(chunk)
            if total - last_print > 200 * 1024 * 1024:
                last_print = total
                pct = total / entry["uncomp_size"] * 100 if entry["uncomp_size"] else 0
                print(f"  {entry['name']}: {total / 1e9:.2f} / {entry['uncomp_size'] / 1e9:.2f} GB ({pct:.1f}%)", flush=True)
        if dec is not None:
            tail = dec.flush()
            f.write(tail)
            total += len(tail)

    tmp.rename(out_path)
    print(f"  -> wrote {out_path} ({total / 1e9:.2f} GB)", flush=True)


def main():
    total_size = http_size(URL)
    print(f"Remote ZIP size: {total_size / 1e9:.2f} GB", flush=True)

    cd_bytes, n_total = parse_central_directory(URL, total_size)
    entries = list(iter_central_entries(cd_bytes))
    print(f"Parsed {len(entries)} entries from central directory", flush=True)

    by_basename = {}
    for e in entries:
        bn = e["name"].rsplit("/", 1)[-1]
        by_basename.setdefault(bn, []).append(e)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    targets = list(TARGETS)
    missing = []
    for tgt in targets:
        if tgt not in by_basename:
            missing.append(tgt)
            continue
        cands = by_basename[tgt]
        entry = sorted(cands, key=lambda x: -x["uncomp_size"])[0]
        print(f"Target {tgt}: archive path = {entry['name']}", flush=True)

        # Decide output: data/cholec80/video01/video01.mp4 if dir already exists,
        # else data/cholec80/videos/video01.mp4
        stem = Path(tgt).stem  # video01
        target_dir = OUT_DIR / stem
        if target_dir.is_dir():
            out_path = target_dir / tgt
        else:
            out_path = OUT_DIR / "videos" / tgt

        if out_path.exists() and out_path.stat().st_size > 100 * 1024 * 1024:
            print(f"  -> exists, skipping ({out_path.stat().st_size / 1e9:.2f} GB)", flush=True)
            continue

        download_entry(URL, entry, out_path)

    if missing:
        print(f"\nNot found in archive: {missing}", flush=True)
        sample = [e["name"] for e in entries if e["name"].lower().endswith(".mp4")][:10]
        print(f"Sample mp4 paths: {sample}", flush=True)


if __name__ == "__main__":
    main()
