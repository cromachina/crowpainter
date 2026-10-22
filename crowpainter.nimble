# Package

version       = "0.0.0"
author        = "cromachina"
description   = "Paint and animation program"
license       = "GPL-3.0-or-later"
srcDir        = "src"
bin           = @["crowpainter"]
binDir        = "build"
backend       = "cpp"

# Dependencies

requires "nim >= 2.2.12"
requires "fusion"
requires "imguin"
requires "sdl3_nim"
requires "parazoa"
requires "patty"
requires "opengl"
requires "zip"