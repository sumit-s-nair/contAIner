# contAIner Python sandbox image (Stage 2G).
#
# python:3.11-slim + git + a C/C++ toolchain (build-essential, pkg-config).  Without git, `pip install .` fails for every repo
# whose build derives its version from VCS metadata (setuptools-scm,
# hatch-vcs, vcs-versioning, versioneer...): dask/dask failed with
# "unable to detect version ... SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DASK".
# A real developer machine has git, so the sandbox should too.
#
# safe.directory '*': /workspace_rw is a `cp -a` of the host bind mount, so the
# files keep the host owner uid; without this git refuses the repo
# ("dubious ownership") and setuptools-scm fails the same way.
#
# build-essential + pkg-config: the --all oracle run (2026-10-15, 36/42) lost
# 4 repos to a missing compiler, not to the repos themselves: ujson (g++,
# jingyaogong/minimind), TgCrypto (gcc, sanjit-sinha/Tg-UsenetBot),
# pycairo/manimpango (gcc, ManimCommunity/manim), PyAV (pkg-config,
# myshell-ai/OpenVoice).  Same argument as git: a developer machine has a
# compiler.  Library -dev packages (libcairo2-dev, libav*-dev ...) are NOT
# added: that would tailor the image to specific repos; repos that still need
# them stay oracle_unwinnable.
#
# Build:  python scripts/build_sandbox_image.py
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git build-essential pkg-config \
 && rm -rf /var/lib/apt/lists/* \
 && git config --system --add safe.directory '*'

LABEL com.contai.sandbox="python-3.11-git-toolchain-v2"
