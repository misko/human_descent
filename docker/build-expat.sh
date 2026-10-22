#!/bin/sh
set -eu

version=2.9.0
source_sha256=16afbb9cefead2aa278105cf27d9f597bde7fbf3dbb85015857ca7ca6a4e89ba
mkdir -p /build/expat /out
cd /build/expat
curl --fail --show-error --silent --location --proto '=https' --tlsv1.2 \
    --retry 3 --output expat.tar.gz \
    "https://github.com/libexpat/libexpat/releases/download/R_2_9_0/expat-${version}.tar.gz"
printf '%s  expat.tar.gz\n' "$source_sha256" | sha256sum -c -
tar xzf expat.tar.gz
cd "expat-${version}"
CFLAGS='-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=3' \
    LDFLAGS='-Wl,-z,relro,-z,now' \
    ./configure --prefix=/usr --libdir="/usr/lib/$(gcc -dumpmachine)" \
        --enable-shared --disable-static --without-docbook --without-xmlwf
make -j4
make check
make DESTDIR=/staging-expat install

architecture=$(dpkg --print-architecture)
library_dir="usr/lib/$(gcc -dumpmachine)"
mkdir -p /package-expat/DEBIAN "/package-expat/$library_dir" \
    /package-expat/usr/share/doc/libexpat1
cp -a "/staging-expat/$library_dir"/libexpat.so.1* "/package-expat/$library_dir/"
cp COPYING /package-expat/usr/share/doc/libexpat1/copyright
cat > /package-expat/DEBIAN/control <<EOF
Package: libexpat1
Version: ${version}-1+security1
Source: expat
Architecture: ${architecture}
Multi-Arch: same
Section: libs
Priority: optional
Maintainer: Local server security maintenance <local-security@localhost>
Depends: libc6 (>= 2.43)
Homepage: https://libexpat.github.io/
Description: Expat upstream security release built from verified source
 Upstream tests run before packaging the runtime library.
EOF
dpkg-deb --root-owner-group --build /package-expat \
    "/out/libexpat1_${version}-1+security1_${architecture}.deb"
printf '{"version":"%s","source_sha256":"%s","upstream_tests":"passed"}\n' \
    "$version" "$source_sha256" > /out/expat-build.json
