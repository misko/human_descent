#!/bin/sh
set -eu

# Verified against the official GitHub release asset digest for pcre2-10.49.
version=10.49
source_sha256=929f0b20e62879252a15886b06c89f1edef61a363cbd5826fb041080a5e557ae
mkdir -p /build /out
cd /build
curl --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
    --retry 3 --output pcre2.tar.gz \
    "https://github.com/PCRE2Project/pcre2/releases/download/pcre2-${version}/pcre2-${version}.tar.gz"
printf '%s  pcre2.tar.gz\n' "$source_sha256" | sha256sum -c -
tar xzf pcre2.tar.gz
cd "pcre2-${version}"
CFLAGS='-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=3' \
    LDFLAGS='-Wl,-z,relro,-z,now' \
    ./configure --prefix=/usr --libdir="/usr/lib/$(gcc -dumpmachine)" \
        --enable-shared --disable-static --enable-jit
make -j4
make check
make DESTDIR=/staging install

architecture=$(dpkg --print-architecture)
library_dir="usr/lib/$(gcc -dumpmachine)"
mkdir -p /package/DEBIAN "/package/$library_dir" \
    /package/usr/share/doc/libpcre2-8-0
cp -a "/staging/$library_dir"/libpcre2-8.so.0* "/package/$library_dir/"
cp LICENCE.md /package/usr/share/doc/libpcre2-8-0/copyright
cp deps/sljit/LICENSE /package/usr/share/doc/libpcre2-8-0/sljit-LICENSE
cat > /package/DEBIAN/control <<EOF
Package: libpcre2-8-0
Version: ${version}-1+security1
Source: pcre2
Architecture: ${architecture}
Multi-Arch: same
Section: libs
Priority: optional
Maintainer: Local server security maintenance <local-security@localhost>
Depends: libc6 (>= 2.43)
Homepage: https://github.com/PCRE2Project/pcre2
Description: PCRE2 upstream security release built from verified source
 Upstream tests run before packaging the runtime library.
EOF
dpkg-deb --root-owner-group --build /package \
    "/out/libpcre2-8-0_${version}-1+security1_${architecture}.deb"
printf '{"version":"%s","source_sha256":"%s","upstream_tests":"passed"}\n' \
    "$version" "$source_sha256" > /out/pcre2-build.json
