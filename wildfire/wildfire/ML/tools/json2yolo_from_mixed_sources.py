import os, json, argparse, random, shutil, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def ensure(p): os.makedirs(p, exist_ok=True)
def list_jsons(d): return sorted([os.path.join(d,f) for f in os.listdir(d) if f.lower().endswith(".json")])

def build_index(roots, workers):
    idx={}
    exts={".jpg",".jpeg",".png",".bmp",".webp",".JPG",".JPEG",".PNG",".BMP",".WebP"}
    def walk_root(r):
        out=[]
        for dp,_,files in os.walk(r):
            for f in files:
                n,e=os.path.splitext(f)
                if e in exts or e.lower() in exts:
                    out.append(os.path.join(dp,f))
        return out
    files=[]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut=[ex.submit(walk_root,r) for r in roots]
        for f in as_completed(fut):
            files.extend(f.result())
    for p in files:
        stem=os.path.splitext(os.path.basename(p))[0].lower()
        if stem not in idx: idx[stem]=p
    return idx

def coco_xyxy_to_yolo(box, W, H):
    x1,y1,x2,y2=box
    w=max(0.0,x2-x1); h=max(0.0,y2-y1)
    xc=(x1+x2)/2.0/W; yc=(y1+y2)/2.0/H; ww=w/W; hh=h/H
    return min(1,max(0,xc)), min(1,max(0,yc)), min(1,max(1e-6,ww)), min(1,max(1e-6,hh))

def map_class(c):
    try: k=int(str(c).strip())
    except: return None
    if k==4: return 0
    if k in (1,2,3): return 1
    return None

def parse_json(fp):
    j=json.load(open(fp,"r",encoding="utf-8"))
    img=j.get("image",{})
    fn=img.get("filename") or img.get("file_name") or img.get("name")
    W=img.get("width"); H=img.get("height")
    if (W is None or H is None):
        res=img.get("resolution")
        if isinstance(res,(list,tuple)) and len(res)>=2: W,H=res[0],res[1]
        elif isinstance(res,dict): W,H=res.get("width"),res.get("height")
    anns=j.get("annotations") or []
    items=[]
    for a in anns:
        cls=map_class(a.get("class") or a.get("category_id") or a.get("label"))
        box=a.get("box") or a.get("bbox")
        if cls is None or not box or not W or not H:
            continue
        if len(box)==4 and box[2]>box[0] and box[3]>box[1]:
            items.append((cls, *coco_xyxy_to_yolo(box,W,H)))
    return fn, items

def write_label(p, items):
    if not items:
        open(p,"w",encoding="utf-8").close()
        return
    with open(p,"w",encoding="utf-8") as f:
        for cls,xc,yc,ww,hh in items:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

def split_indices(n, ratios, seed):
    idx=list(range(n))
    random.Random(seed).shuffle(idx)
    a=int(n*ratios[0]); b=int(n*ratios[1])
    return set(idx[:a]), set(idx[a:a+b]), set(idx[a+b:])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True)
    ap.add_argument("--images_dirs", nargs="+", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--ratio", default="0.7,0.2,0.1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy_mode", choices=["copy","link"], default="copy")
    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8)))
    args=ap.parse_args()

    ratios=tuple(float(x) for x in args.ratio.split(","))
    out_imgs=os.path.join(args.out_root,"images")
    out_lbls=os.path.join(args.out_root,"labels")
    for s in ["train","val","test"]:
        ensure(os.path.join(out_imgs,s)); ensure(os.path.join(out_lbls,s))

    json_files=list_jsons(args.json_dir)
    print(f"found json: {len(json_files)} in {args.json_dir}")
    if not json_files:
        print("prepared: 0 images"); return

    idx=build_index(args.images_dirs, args.workers)
    print(f"indexed images: {len(idx)} stems")

    paired=[]; missing=[]
    def parse_and_match(fp):
        try:
            fn, items=parse_json(fp)
        except Exception as e:
            return ("err", os.path.basename(fp), str(e))
        if not fn:
            return ("miss", os.path.basename(fp), "no-filename")
        stem=os.path.splitext(fn)[0].lower()
        ip=idx.get(stem)
        if not ip:
            return ("miss", os.path.basename(fp), fn)
        return ("ok", ip, items)

    results=[]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        fut=[ex.submit(parse_and_match, jf) for jf in json_files]
        for f in as_completed(fut):
            results.append(f.result())

    for tag,a,b in results:
        if tag=="ok": paired.append((a,b))
        elif tag=="miss": missing.append((a,b))
        else: missing.append((a,b))

    n=len(paired)
    tr,va,te=split_indices(n, ratios, args.seed)

    def save_one(i_ip_items):
        i,(ip,items)=i_ip_items
        split="train" if i in tr else "val" if i in va else "test"
        dst_img=os.path.join(out_imgs,split,os.path.basename(ip))
        dst_lbl=os.path.join(out_lbls,split,os.path.splitext(os.path.basename(ip))[0]+".txt")
        if args.copy_mode=="copy":
            shutil.copy2(ip, dst_img)
        else:
            if not os.path.exists(dst_img):
                os.link(ip, dst_img) if sys.platform!="win32" else shutil.copy2(ip, dst_img)
        write_label(dst_lbl, items)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(save_one, enumerate(paired)))

    if missing:
        mp=os.path.join(args.out_root,"_missing_images.txt")
        with open(mp,"w",encoding="utf-8") as f:
            for j,fn in missing: f.write(f"{j}\t{fn}\n")
        print("missing:", len(missing), "->", mp)
    print("prepared:", n, "images")

if __name__=="__main__":
    main()