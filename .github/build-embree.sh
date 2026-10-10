#!/usr/bin/env bash
# Builds Embree from source into ./embree as static libraries, for the wheels to carry: triangles and intersect
# filters only, its own tasking in place of TBB, single rays. build.rs links whatever is installed there.
set -euo pipefail
VERSION=4.4.0
PREFIX="$(pwd)/embree"
test -e "$PREFIX/include/embree4/rtcore.h" && exit 0
SOURCE="$(mktemp -d)"
curl -sSL "https://github.com/RenderKit/embree/archive/refs/tags/v$VERSION.tar.gz" | tar -xz -C "$SOURCE" --strip-components=1
case "$(uname -m)" in
  x86_64|AMD64) options=(-DEMBREE_MAX_ISA=AVX2) ;;
  *) options=() ;;  # ARM: NEON, Embree's default there
esac
case "$(uname -s)" in
  MINGW*|MSYS*) options+=(-DCMAKE_CXX_FLAGS_INIT=/MP -DCMAKE_C_FLAGS_INIT=/MP) ;;  # MSVC compiles a project's files one at a time unless asked
esac
cmake -S "$SOURCE" -B "$SOURCE/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 ${options[@]+"${options[@]}"} \
  -DEMBREE_STATIC_LIB=ON -DEMBREE_TASKING_SYSTEM=INTERNAL -DEMBREE_ISPC_SUPPORT=OFF -DEMBREE_TUTORIALS=OFF \
  -DEMBREE_RAY_PACKETS=OFF -DEMBREE_FILTER_FUNCTION=ON -DEMBREE_GEOMETRY_TRIANGLE=ON -DEMBREE_GEOMETRY_QUAD=OFF \
  -DEMBREE_GEOMETRY_CURVE=OFF -DEMBREE_GEOMETRY_SUBDIVISION=OFF -DEMBREE_GEOMETRY_USER=OFF \
  -DEMBREE_GEOMETRY_INSTANCE=OFF -DEMBREE_GEOMETRY_INSTANCE_ARRAY=OFF -DEMBREE_GEOMETRY_GRID=OFF -DEMBREE_GEOMETRY_POINT=OFF
cmake --build "$SOURCE/build" --config Release --parallel 4
cmake --install "$SOURCE/build" --config Release
