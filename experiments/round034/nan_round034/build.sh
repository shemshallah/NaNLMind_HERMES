#!/bin/sh
set -eu
c++ -std=c++20 -O3 -fPIC -shared native_episodic.cpp -o libepisodic.so
