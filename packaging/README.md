# Packaging

Self-contained builds are made with PyInstaller from `connflex.spec`:

```sh
pip install -e ".[build]"
pyinstaller packaging/connflex.spec --noconfirm --clean
```

The result lands in `dist/`: `ConnFlex.app` on macOS, a `ConnFlex/` folder with
`ConnFlex.exe` on Windows, and a `ConnFlex/` folder with a `ConnFlex` binary on Linux.
Verify a build with the bundled checks:

```sh
dist/ConnFlex.app/Contents/MacOS/ConnFlex --self-test     # headless pipeline on the demo
dist/ConnFlex.app/Contents/MacOS/ConnFlex --smoke-test    # real GUI, saves smoke_test.png
```

`make_icons.py` regenerates `icon.png`, `icon.ico`, and `icon.icns`.

The bundle's version comes from the git tag through setuptools-scm (run the build from a
checkout with tags fetched). Apple's `CFBundleShortVersionString` gets the plain `X.Y.Z`
part; the full version string is stored under the `ConnFlexVersion` key of `Info.plist`.

The GitHub release workflow (`.github/workflows/release.yml`) runs these steps on Linux,
Windows, and Apple-silicon macOS runners whenever a `v*` tag is pushed, and attaches the
archives to a GitHub release. The macOS app is ad-hoc signed, not notarized: on first launch,
right-click the app and choose Open, or clear the quarantine flag with
`xattr -dr com.apple.quarantine ConnFlex.app`.
