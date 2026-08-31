cmake_minimum_required(VERSION 3.16)

# Phase 2 intentionally has no compiled sources yet. Keep a named target so
# robot-specific deploy packages can link the common seam without special
# casing an empty directory.
add_library(sonic_common INTERFACE)
target_compile_features(sonic_common INTERFACE cxx_std_20)

