"""
geo_stream.py  (CNS task - helpers to stream GEO series files)
=============================================================
Two helpers used by build_external_cns.py: the GEO FTP directory of a series, and a text stream of a gzipped file
read over HTTP (no local copy).
"""
import gzip
import io
import urllib.request

GEO = "https://ftp.ncbi.nlm.nih.gov/geo/series"


def series_dir(gse):
    n = gse[3:]
    return f"{GEO}/GSE{n[:-3]}nnn/{gse}"


def open_stream(url, timeout=120):
    return io.TextIOWrapper(gzip.GzipFile(fileobj=urllib.request.urlopen(url, timeout=timeout)),
                            encoding="utf-8", errors="replace")
