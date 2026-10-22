#!/bin/sh
set -eu

version=0.9.13
# Published by the upstream maintainer with the release announcement.
source_sha256=bf264b683d1e49ff9a6bc03af13baa430a556d943731a2b1ac2439dae4fea205
mkdir -p /build/xrender /out-xrender
cd /build/xrender
python -c "import urllib.request; urllib.request.urlretrieve('https://xorg.freedesktop.org/archive/individual/lib/libXrender-${version}.tar.gz', 'source.tar.gz')"
printf '%s  source.tar.gz\n' "$source_sha256" | sha256sum -c -
tar xzf source.tar.gz
cd "libXrender-${version}"
CFLAGS='-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=3' \
    LDFLAGS='-Wl,-z,relro,-z,now' \
    ./configure --prefix=/usr --libdir="/usr/lib/$(gcc -dumpmachine)" \
        --enable-shared --disable-static
make -j4
make check
make DESTDIR=/staging-xrender install

architecture=$(dpkg --print-architecture)
library_dir="usr/lib/$(gcc -dumpmachine)"
mkdir -p /package-xrender/DEBIAN "/package-xrender/$library_dir" \
    /package-xrender/usr/share/doc/libxrender1
cp -a "/staging-xrender/$library_dir"/libXrender.so.1* "/package-xrender/$library_dir/"
cp COPYING /package-xrender/usr/share/doc/libxrender1/copyright
cat > /package-xrender/DEBIAN/control <<EOF
Package: libxrender1
Version: 1:${version}-1+security1
Source: libxrender
Architecture: ${architecture}
Multi-Arch: same
Section: libs
Priority: optional
Maintainer: Local server security maintenance <local-security@localhost>
Depends: libc6 (>= 2.43), libx11-6 (>= 2:1.8)
Homepage: https://xorg.freedesktop.org/
Description: Xrender upstream security release built from verified source
 Fixes CVE-2026-88807 in the X11 reply parser.
EOF
dpkg-deb --root-owner-group --build /package-xrender \
    "/out-xrender/libxrender1_${version}-1+security1_${architecture}.deb"
printf '{"version":"%s","source_sha256":"%s","upstream_make_check":"completed","fixes":["CVE-2026-88807"]}\n' \
    "$version" "$source_sha256" > /out-xrender/xrender-build.json
