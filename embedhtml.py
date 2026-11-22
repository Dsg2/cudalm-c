# monitor.html -> monitor_html.h (one C string literal per line; MSVC caps a literal at 64 KB total)
import sys, os
d = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(d, "monitor.html"), encoding="utf-8").read().replace("\r\n", "\n")
b = src.encode("utf-8")
if len(b) > 60000: sys.exit("monitor.html is %d bytes; MSVC limit is ~64 KB" % len(b))
out = ["/* generated from monitor.html by embedhtml.py - edit that file, not this one */", "static const char g_mhtml[] ="]
lines = src.split("\n")
for i, ln in enumerate(lines):
    if i == len(lines) - 1 and not ln: break
    nl = "\\n" if i < len(lines) - 1 else ""
    s = ln.replace("\\", "\\\\").replace('"', '\\"').replace("??", "?\\?")
    s = "".join(c if ord(c) < 128 else "".join("\\%03o" % x for x in c.encode("utf-8")) for c in s)
    out.append('"%s%s"' % (s, nl))
out[-1] += ";"
open(os.path.join(d, "monitor_html.h"), "w", encoding="ascii", newline="\n").write("\n".join(out) + "\n")
print("monitor_html.h: %d bytes of html" % len(b))
