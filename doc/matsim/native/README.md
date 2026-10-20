# Native network reader and shared cache

Use `read_network(filename, context=context)`. All pipeline callers pass their
Synpp context. Without a context the API still reads networks, but does not
persist a binary network cache.

Binary files are stored as `context.working_directory/network_<sha256>.pkl`.
The stable key includes the resolved absolute XML path, nanosecond modification
time, file size, `skip_attributes`, and the cache schema version. Updating a
source produces a new key; old cache files are not automatically deleted.
`only_car_links` is applied after caching so it cannot contaminate the full
network. Each read returns independent tables and dictionaries.

The first cache miss uses the C++ streaming reader, builds the same `Network`
object and pandas tables, and writes the pickle atomically. A per-key file lock
prevents concurrent readers from parsing the same input twice. Corrupt cache
files are rebuilt. If the source changes during parsing, the read fails without
publishing a cache file.

The extension is compiled automatically on first use under
`context.working_directory/network_native/<source-and-Python-hash>/`. It needs a
C++17 compiler (`CXX`, or `g++`), Python headers, Expat headers/library, and zlib
headers/library. On systems without these, a warning is logged and the Python
reader is used; binary caching still works. Changes to the C++ source or Python
ABI produce a separate extension build. On standalone reads, extension builds
use a per-user directory under the system temporary directory.

The native parser accepts plain or gzipped XML, namespaces, elevations, network
CRS, node/link attributes and turn restrictions. IDs remain strings; typed
integer attributes retain integer precision. Other compressed formats use the
existing xopen/Python path. Numeric/XML/gzip errors propagate rather than being
silently converted to missing or zero values. Both readers preserve the public
`nodes`, `links`, `node_attrs`, `link_attrs`, and `network_attrs` interface.

First reads include compilation when needed, XML parsing, and writing the binary
cache; they are not guaranteed to finish within 30 seconds. Subsequent reads
skip both compilation and XML parsing. Cache files use the project's Python
`Network` class; increment `CACHE_VERSION` in `matsim/network_reader.py` when
changing the cached schema or parsing semantics.

## Integrated measurements (2026-10-18)

On the generated prepared networks, with the shared cache on project storage:

| Links | First read including cache write | Cached read |
|---:|---:|---:|
| 1,603,777 | 37.8 s (also includes first compilation) | 5.1 s |
| 2,915,180 | 65.2 s | 7.9 s |

Measurements exclude Python imports. All four tables and network metadata were
compared exactly against the benchmark reference for each network. Cache reads
were measured with warm filesystem caches; the complete simulation pipeline was
not executed. The reader tests cover parsing parity, invalidation, independent
filtered reads, concurrency, malformed files, compiler fallback, and atomic
cache-write failure handling.
