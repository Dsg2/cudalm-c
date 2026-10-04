#!/usr/bin/env python3
"""build.py -- one-step build of cudalm (and its helper tools) with automatic toolchain download.

    python build.py                 build cudalm.exe next to this file
    python build.py --tools         also build judge.exe / teachgen.exe (needs gcc, or MSVC)
    python build.py --python-deps   also pip-install what prep/fetch.py needs (pyarrow)
    python build.py --dry-run       show what would be downloaded / run, change nothing

What it does:
  1. CUDA: uses --cuda DIR, else a suitable installed toolkit (CUDA_PATH / nvcc on PATH, >= 12.8 for sm_120),
     else downloads the minimal parts (nvcc, cudart, cccl, cublas; ~650 MB) from NVIDIA's redistributable
     archives into .deps/cuda-<ver>/ (sha256-checked, reused on later builds).
  2. Host compiler: Windows needs MSVC. An installed Visual Studio 2022 / Build Tools is found via vswhere; without
     one, the x64 compiler + Windows SDK (~290 MB download) are unpacked from Microsoft's package feed into .deps/msvc
     (no install, no admin; asks you to accept the Build Tools license, or pass --accept-msvc-license).
     --install-msvc instead runs Microsoft's Build Tools installer (admin, several GB).
     Linux uses g++ from PATH (Linux builds are untested; `cudalm data` is Windows-only).
  3. GPU arch: --arch (e.g. 120 / 89 / 86), default = what nvidia-smi reports, else 120. Only sm_120 is tested.
  4. nvcc -O3 ... -> cudalm(.exe); on Windows the cuBLAS DLLs are copied next to the exe.
"""
import argparse, glob, hashlib, json, os, platform, re, shutil, subprocess, sys, tarfile, urllib.parse, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEPS = os.path.join(HERE, ".deps")
WIN = os.name == "nt"
REDIST = "https://developer.download.nvidia.com/compute/cuda/redist/"
PARTS = ["cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas"]
EXTRA = ["cuda_crt", "libnvvm"]            # split out of cuda_nvcc in CUDA 13
VS_URL = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
VS_CHANNEL = "https://aka.ms/vs/17/release/channel"

for s in (sys.stdout, sys.stderr): s.reconfigure(errors="replace")  # localized OS errors on a cp1252 console
def log(*a): print("[build]", *a, flush=True)
def die(msg): print("[build] ERROR:", msg, file=sys.stderr); sys.exit(1)

def run(cmd, env=None, dry=False, **kw):
    log(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry: return 0
    return subprocess.call(cmd, env=env, **kw)

def fetch(url, dst, sha=None, dry=False):
    sha = sha and sha.lower()
    if os.path.exists(dst) and (not sha or sha256(dst) == sha): return dst
    if dry: log("would download", url); return dst
    log("downloading", url)
    tmp = dst + ".part"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                n, tot, last = 0, int(r.headers.get("Content-Length") or 0), -1
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); n += len(b)
                    if tot > 2**22 and n * 20 // tot != last: last = n * 20 // tot; print(f"\r  {n/2**20:7.0f} / {tot/2**20:.0f} MB", end="", flush=True)
            if tot > 2**22: print()
            break
        except OSError as e:
            if attempt == 4: die(f"download failed: {url} ({e})")
            log(f"download interrupted ({e}), retrying")
    if sha and sha256(tmp) != sha: os.remove(tmp); die(f"checksum mismatch for {url}")
    os.replace(tmp, dst); return dst

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------- CUDA
def nvcc_version(root):
    nv = os.path.join(root, "bin", "nvcc.exe" if WIN else "nvcc")
    if not os.path.isfile(nv): return None
    try: out = subprocess.run([nv, "--version"], capture_output=True, text=True).stdout
    except OSError: return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m[1]), int(m[2])) if m else None

def has_cublas(root):
    lib = os.path.join(root, "lib", "x64", "cublas.lib") if WIN else os.path.join(root, "lib64", "libcublas.so")
    return os.path.exists(lib) or os.path.exists(os.path.join(root, "lib", "libcublas.so"))

def find_cuda(need):
    cands = [os.environ.get("CUDA_PATH"), os.environ.get("CUDA_HOME")]
    nv = shutil.which("nvcc")
    if nv: cands.append(os.path.dirname(os.path.dirname(nv)))
    if not WIN: cands.append("/usr/local/cuda")
    for c in filter(None, cands):
        v = nvcc_version(c)
        if v and v >= need and has_cublas(c): return c, v
    return None, None

def download_cuda(ver, dry):
    plat = "windows-x86_64" if WIN else "linux-x86_64" if platform.machine() in ("x86_64", "AMD64") else "linux-sbsa"
    root = os.path.join(DEPS, f"cuda-{ver}")
    if os.path.exists(os.path.join(root, ".complete")): return root
    if dry:
        with urllib.request.urlopen(REDIST + f"redistrib_{ver}.json") as r: m = json.load(r)
    else:
        os.makedirs(os.path.join(DEPS, "dl"), exist_ok=True)
        m = json.load(open(fetch(REDIST + f"redistrib_{ver}.json", os.path.join(DEPS, "dl", f"redistrib_{ver}.json"))))
    parts = PARTS + [p for p in EXTRA if p in m]
    for p in parts:
        if p not in m or plat not in m[p]: die(f"{p} has no {plat} archive in CUDA {ver}")
        a = m[p][plat]
        log(f"{p} {m[p]['version']}  ({int(a.get('size', 0))/2**20:.0f} MB)")
        arc = fetch(REDIST + a["relative_path"], os.path.join(DEPS, "dl", os.path.basename(a["relative_path"])), a["sha256"], dry)
        if not dry: extract(arc, root)
    if not dry:
        open(os.path.join(root, ".complete"), "w").write(ver)
        shutil.rmtree(os.path.join(DEPS, "dl"), ignore_errors=True)
        log("CUDA ready in", root)
    return root

def extract(arc, root):
    """unpack an NVIDIA redist archive into root, dropping its top-level folder"""
    log("extracting", os.path.basename(arc))
    if arc.endswith(".zip"):
        with zipfile.ZipFile(arc) as z:
            for i in z.infolist():
                rel = i.filename.split("/", 1)[1] if "/" in i.filename else ""
                if not rel or i.is_dir(): continue
                dst = os.path.join(root, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
                with z.open(i) as s, open(dst, "wb") as d: shutil.copyfileobj(s, d)
    else:
        with tarfile.open(arc) as t:
            for i in t.getmembers():
                rel = i.name.split("/", 1)[1] if "/" in i.name else ""
                if not rel: continue
                i.name = rel; t.extract(i, root, filter="tar") if hasattr(tarfile, "data_filter") else t.extract(i, root)

# ---------------------------------------------------------------- host compiler
def vswhere():
    p = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"), r"Microsoft Visual Studio\Installer\vswhere.exe")
    if not os.path.exists(p): return None
    out = subprocess.run([p, "-latest", "-products", "*", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                          "-property", "installationPath"], capture_output=True, text=True).stdout.strip()
    bat = os.path.join(out, r"VC\Auxiliary\Build\vcvars64.bat") if out else ""
    return bat if os.path.exists(bat) else None

def msvc_env(a, dry):
    bat = vswhere()
    if not bat and a.install_msvc:
        exe = fetch(VS_URL, os.path.join(DEPS, "vs_BuildTools.exe"), dry=dry)
        log("installing Visual Studio Build Tools (C++ workload) -- accept the UAC prompt")
        run([exe, "--passive", "--wait", "--norestart", "--add", "Microsoft.VisualStudio.Workload.VCTools", "--includeRecommended"], dry=dry)
        bat = vswhere()
    if not bat: return portable_msvc(a.accept_msvc_license, dry)
    log("MSVC:", bat)
    out = subprocess.run(f'cmd /s /c ""{bat}" >nul && set"', capture_output=True, text=True, shell=True).stdout
    env = {k.upper(): v for k, v in (l.split("=", 1) for l in out.splitlines() if "=" in l)}
    env["VSLANG"] = "1033"                           # English compiler messages
    return env

def getjson(url):
    with urllib.request.urlopen(url) as r: return json.load(r)

def portable_msvc(accept, dry):
    """no Visual Studio: unpack just the x64 compiler + Windows SDK from Microsoft's VS package feed into .deps/msvc"""
    root = os.path.join(DEPS, "msvc")
    if not os.path.exists(os.path.join(root, ".complete")):
        ch = getjson(VS_CHANNEL)
        lic = next(r["license"] for i in ch["channelItems"] if i["id"] == "Microsoft.VisualStudio.Product.BuildTools"
                   for r in i.get("localizedResources", []) if r["language"].lower() == "en-us")
        if dry: log(f"MSVC not found: would download the portable compiler + Windows SDK into {root} (license: {lic})"); return dict(os.environ)
        if not accept:
            q = ("MSVC not found. Download Microsoft's C++ compiler + Windows SDK (~290 MB download, ~750 MB unpacked, no install, no admin) into .deps/msvc?\n"
                 f"        This accepts the Visual Studio Build Tools license: {lic}\n        [y/N] ")
            if not sys.stdin.isatty() or input("[build] " + q).strip().lower() != "y":
                die("MSVC is required: re-run with --accept-msvc-license, or install VS 2022 Build Tools (C++ workload) / --install-msvc")
        vs = getjson(next(i for i in ch["channelItems"] if i["id"] == "Microsoft.VisualStudio.Manifests.VisualStudio")["payloads"][0]["url"])
        pk = {}
        for p in vs["packages"]: pk.setdefault(p["id"].lower(), []).append(p)
        def pick(pid): return next((p for p in pk.get(pid, []) if p.get("language") in (None, "en-US")), None)
        vers = [i.split(".")[4:-2] for i in pk if i.startswith("microsoft.visualstudio.component.vc.") and i.endswith(".x86.x64")]
        mv = ".".join(max((v for v in vers if v and all(x.isdigit() for x in v)), key=lambda v: [int(x) for x in v]))
        sdks = [i for i in pk if re.match(r"microsoft\.visualstudio\.component\.windows1[01]sdk\.\d+$", i)]
        sdk = pick(max(sdks, key=lambda i: int(i.split(".")[-1])))
        sdk = pick(next(iter(sdk["dependencies"])).lower())
        dl = os.path.join(DEPS, "dl-msvc"); os.makedirs(dl, exist_ok=True)
        log(f"portable MSVC {mv} + {sdk['id']}")
        for name in ["crt.headers.base", "tools.hostx64.targetx64.base", "tools.hostx64.targetx64.res.base", "crt.x64.desktop.base",
                     "crt.x64.store.base"]:
            for pl in pick(f"microsoft.vc.{mv}.{name}")["payloads"]:
                f = fetch(pl["url"], os.path.join(dl, pl["sha256"][:16].lower() + ".vsix"), pl["sha256"])
                log("extracting", pl["fileName"])
                with zipfile.ZipFile(f) as z:
                    for i in z.infolist():
                        n = i.filename.lower()
                        if not n.startswith("contents/") or i.is_dir() or n.endswith((".pdb", "vctip.exe")) or re.search(r"/x64/(store|uwp|enclave|onecore)/", n): continue
                        d = os.path.join(root, urllib.parse.unquote(i.filename[9:])); os.makedirs(os.path.dirname(d), exist_ok=True)
                        with z.open(i) as src, open(d, "wb") as out: shutil.copyfileobj(src, out)
        msis = ["Windows SDK for Windows Store Apps Headers", "Windows SDK for Windows Store Apps Headers OnecoreUap", "Windows SDK for Windows Store Apps Libs", "Universal CRT Headers Libraries and Sources",
                "Windows SDK Desktop Headers x64", "Windows SDK Desktop Headers x86", "Windows SDK Desktop Libs x64"]
        byname = {pl["fileName"]: pl for pl in sdk["payloads"]}
        for m in msis:
            pl = byname[f"Installers\\{m}-x86_en-us.msi"]
            f = fetch(pl["url"], os.path.join(dl, m + ".msi"), pl["sha256"])
            data = open(f, "rb").read()
            for k in {x.start() for x in re.finditer(rb"\.cab", data)}:
                cab = f"Installers\\{data[k - 32:k + 4].decode('ascii', 'replace')}"
                if cab in byname: fetch(byname[cab]["url"], os.path.join(dl, cab[11:]), byname[cab]["sha256"])
            log("unpacking", m)
            if subprocess.call(["msiexec.exe", "/a", f, "/quiet", "/qn", "TARGETDIR=" + os.path.abspath(root)]): die(f"msiexec failed on {m}")
            if os.path.exists(os.path.join(root, m + ".msi")): os.remove(os.path.join(root, m + ".msi"))
        kit = os.path.join(root, "Windows Kits", "10")      # the msis carry every arch: keep x64 only
        for d in (glob.glob(os.path.join(kit, "Lib", "*", "*", "*")) + glob.glob(os.path.join(kit, "bin", "*", "*")) +
                  glob.glob(os.path.join(kit, "Include", "*", "cppwinrt")) + [os.path.join(kit, "Source")]):
            if os.path.basename(d).lower() != "x64": shutil.rmtree(d, ignore_errors=True)
        aux = os.path.join(root, "VC", "Auxiliary", "Build"); os.makedirs(aux, exist_ok=True)
        for b in ("vcvars64.bat", "vcvarsall.bat"): open(os.path.join(aux, b), "w").write("@echo off\n")  # nvcc checks these exist
        open(os.path.join(root, ".complete"), "w").write(mv)
        shutil.rmtree(dl, ignore_errors=True)
    vc = os.path.join(root, "VC", "Tools", "MSVC"); vc = os.path.join(vc, max(os.listdir(vc)))
    kit = os.path.join(root, "Windows Kits", "10"); sv = max(os.listdir(os.path.join(kit, "Include")))
    log("MSVC (portable):", vc)
    env = {k.upper(): v for k, v in os.environ.items()}
    env["PATH"] = os.pathsep.join([os.path.join(vc, "bin", "Hostx64", "x64"), os.path.join(kit, "bin", sv, "x64"), env.get("PATH", "")])
    env["INCLUDE"] = os.pathsep.join([os.path.join(vc, "include")] + [os.path.join(kit, "Include", sv, d) for d in ("ucrt", "shared", "um", "winrt")])
    env["LIB"] = os.pathsep.join([os.path.join(vc, "lib", "x64")] + [os.path.join(kit, "Lib", sv, d, "x64") for d in ("ucrt", "um")])
    env["VSLANG"] = "1033"
    return env

def gpu_arch():
    try: out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired): return None
    m = re.search(r"(\d+)\.(\d+)", out)
    return m[1] + m[2] if m else None

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cuda", help="use this CUDA toolkit folder (bin/nvcc, include, lib)")
    ap.add_argument("--cuda-version", default="12.9.1", help="redist version to download (default 12.9.1)")
    ap.add_argument("--download-cuda", action="store_true", help="download CUDA even if one is installed")
    ap.add_argument("--arch", default="auto", help="GPU arch, e.g. 120 (RTX 50xx), 89 (RTX 40xx); default: detect")
    ap.add_argument("--out", default="cudalm", help="exe name (default cudalm)")
    ap.add_argument("--tools", action="store_true", help="also build judge and teachgen (bench/, teach/)")
    ap.add_argument("--python-deps", action="store_true", help="pip install requirements.txt")
    ap.add_argument("--install-msvc", action="store_true", help="download + install VS Build Tools if MSVC is missing")
    ap.add_argument("--accept-msvc-license", action="store_true", help="no Visual Studio: fetch the portable compiler without asking")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(); dry = a.dry_run

    det = gpu_arch()
    arch = a.arch if a.arch != "auto" else (det or "120")
    if a.arch == "auto": log(f"GPU arch sm_{arch}" + ("" if det else " (default)"))
    if not det: log("warning: no NVIDIA GPU found (nvidia-smi) -- cudalm will build but needs an NVIDIA GPU + driver to run")
    if arch != "120": log(f"note: only sm_120 has been tested; sm_{arch} builds are best effort")
    need = (12, 8) if int(arch) >= 120 else (12, 0)

    # host compiler first, so a missing MSVC fails before the CUDA download
    env = msvc_env(a, dry) if WIN else dict(os.environ)
    if WIN: env = {k.upper(): v for k, v in env.items()}
    if not WIN and not shutil.which("g++"): die("g++ not found (install build-essential)")
    if a.cuda:
        cuda = a.cuda
        if not nvcc_version(cuda): die(f"no bin/nvcc in {cuda}")
    else:
        cuda, v = (None, None) if a.download_cuda else find_cuda(need)
        if cuda: log(f"CUDA {v[0]}.{v[1]}: {cuda}")
        else: cuda = download_cuda(a.cuda_version, dry)

    env["PATH"] = os.path.join(cuda, "bin") + os.pathsep + env.get("PATH", "")
    nvcc = os.path.join(cuda, "bin", "nvcc.exe" if WIN else "nvcc")
    exe = os.path.join(HERE, a.out + (".exe" if WIN else ""))
    lib = os.path.join(cuda, "lib", "x64") if WIN else os.path.join(cuda, "lib64" if os.path.isdir(os.path.join(cuda, "lib64")) else "lib")
    cmd = [nvcc, "-O3", f"-arch=sm_{arch}", "-diag-suppress", "177,221,550", "-cudart", "static", "-I" + os.path.join(cuda, "include"), "-L" + lib,
           "-o", exe, os.path.join(HERE, "cudalm.cu"), "-lcublas", "-lcublasLt"]
    cmd += ["-allow-unsupported-compiler", "-ladvapi32"] if WIN else ["-Xlinker", "-rpath=" + lib]
    log("compiling cudalm (a few minutes)")
    if run(cmd, env=env, dry=dry, cwd=HERE): die("nvcc failed (see the messages above)")
    if WIN and not dry:
        for f in os.listdir(os.path.join(cuda, "bin")):
            if re.match(r"cublas(Lt)?64_\d+\.dll$", f, re.I):
                d = os.path.join(HERE, f)
                if not os.path.exists(d) or os.path.getsize(d) != os.path.getsize(os.path.join(cuda, "bin", f)):
                    log("copying", f); shutil.copy2(os.path.join(cuda, "bin", f), d)
        for x in (".lib", ".exp"):
            if os.path.exists(exe[:-4] + x): os.remove(exe[:-4] + x)
    log("built", exe)

    if a.tools: build_tools(env, dry)
    if a.python_deps: run([sys.executable, "-m", "pip", "install", "-r", os.path.join(HERE, "requirements.txt")], dry=dry)
    if not dry:
        for d in ("data", "models"): os.makedirs(os.path.join(HERE, d), exist_ok=True)
    log("done. Run it with no arguments for the menu, or `cudalm help`.")

def build_tools(env, dry):
    if not WIN: log("judge/teachgen are Windows-only; skipped"); return
    srcs = [os.path.join(HERE, "bench", "judge.c"), os.path.join(HERE, "teach", "teachgen.c")]
    gcc, cl = shutil.which("gcc"), shutil.which("cl", path=env.get("PATH"))
    if not gcc and not cl: log("no gcc or cl found; tools skipped"); return
    for s in srcs:
        exe = os.path.join(HERE, os.path.splitext(os.path.basename(s))[0] + ".exe")
        cmd = [gcc, "-O2", "-static", "-o", exe, s, "-lws2_32"] if gcc else [cl, "/nologo", "/O2", f"/Fe:{exe}", f"/Fo:{exe}.obj", s, "ws2_32.lib"]
        if run(cmd, env=env, dry=dry, cwd=HERE): die(f"building {os.path.basename(s)} failed")
        if not gcc and os.path.exists(exe + ".obj"): os.remove(exe + ".obj")

if __name__ == "__main__":
    main()
