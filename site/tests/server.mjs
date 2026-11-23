// Tiny static server for page tests. `block` is a set of URL paths that return 404.
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml" };

export function serve(root, block = new Set()) {
  const server = createServer(async (req, res) => {
    const path = decodeURIComponent(new URL(req.url, "http://x").pathname);
    if (block.has(path)) { res.writeHead(404); return res.end("not found"); }
    try {
      const file = normalize(join(root, path.endsWith("/") ? path + "index.html" : path));
      if (!file.startsWith(root)) throw new Error("outside root");
      const body = await readFile(file);
      res.writeHead(200, { "content-type": TYPES[extname(file)] || "application/octet-stream" });
      res.end(body);
    } catch { res.writeHead(404); res.end("not found"); }
  });
  return new Promise((ok) => server.listen(0, () => ok({ server, url: `http://127.0.0.1:${server.address().port}` })));
}
