# -*- coding: utf-8 -*-
"""QC-first v2 runner built on measure_guides.py at commit 8cfae03.

Changes: ruler/header exclusion, conservative touching-corolla splitting, C-prefixed
flower labels, and broad detached reproductive-organ candidate measurement.
"""
from __future__ import annotations
import argparse, csv, math, os
from pathlib import Path
import cv2, numpy as np
import measure_guides as base

SPLIT_LEN_MM = 55.0
SPLIT_AREA_MM2 = 1350.0


def specimen_top(img):
    h, w = img.shape[:2]
    edges = cv2.Canny(cv2.cvtColor(img[:int(h*.48)], cv2.COLOR_BGR2GRAY), 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, max(80, w//8),
                            minLineLength=int(w*.48), maxLineGap=int(w*.06))
    ys=[]
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            if abs(y2-y1) <= max(4,int(h*.004)) and int(h*.12) <= (y1+y2)//2 <= int(h*.46):
                ys.append((y1+y2)//2)
    return min(h-1, (max(ys) if ys else int(h*.28)) + max(24,int(h*.018)))


def foreground_v2(img, top):
    lc,a,b=base.channels(img); chroma=np.sqrt(a*a+b*b); tex=base._lstd(lc,15)
    fg=((chroma>10)|((tex>7)&(chroma>4))).astype(np.uint8)*255; fg[:top]=0
    fg=cv2.morphologyEx(fg,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    fg=cv2.morphologyEx(fg,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(27,27)))
    h,w=fg.shape; n,lab,st,_=cv2.connectedComponentsWithStats(fg,8); out=np.zeros_like(fg)
    for i in range(1,n):
        if st[i,cv2.CC_STAT_AREA]*base.MM2_PX < 20: continue
        c=(lab==i).astype(np.uint8); ff=c.copy()*255
        cv2.floodFill(ff,np.zeros((h+2,w+2),np.uint8),(0,0),255)
        out |= cv2.bitwise_or(c*255,cv2.bitwise_not(ff))
    out[:top]=0
    return out,a,b


def metrics(mask):
    cs,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cs: return None
    c=max(cs,key=cv2.contourArea); rw,rh=cv2.minAreaRect(c)[1]
    hull=cv2.contourArea(cv2.convexHull(c)); area=float(mask.sum())
    return dict(contour=c,area_px=area,length_px=max(rw,rh),width_px=min(rw,rh),
                solidity=area/hull if hull else 0,aspect=max(rw,rh)/max(min(rw,rh),1e-6))


def try_split(mask):
    m=metrics(mask); area=m['area_px']*base.MM2_PX; length=m['length_px']*base.MM_PX
    if length <= SPLIT_LEN_MM and area <= SPLIT_AREA_MM2: return [mask],'not_triggered'
    ys,xs=np.where(mask); pts=np.c_[xs,ys].astype(np.float32)
    _,labs,centers=cv2.kmeans(pts,2,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,60,.5),
                              10,cv2.KMEANS_PP_CENTERS)
    if np.linalg.norm(centers[0]-centers[1]) < max(80,.22*m['length_px']):
        return [mask],'split_centres_too_close'
    children=[]
    for k in (0,1):
        c=np.zeros_like(mask,np.uint8); q=labs.ravel()==k; c[ys[q],xs[q]]=1
        c=cv2.morphologyEx(c,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
        n,lab,st,_=cv2.connectedComponentsWithStats(c,8)
        if n>1: c=(lab==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
        children.append(c)
    cm=[metrics(c) for c in children]; aa=[x['area_px']*base.MM2_PX for x in cm]
    ll=[x['length_px']*base.MM_PX for x in cm]
    ok=all(base.AREA_MM2_MIN<=a<=1500 and 15<=l<=60 and x['solidity']>=.40 and x['aspect']<=4.5
           for a,l,x in zip(aa,ll,cm))
    if ok and min(aa)/max(aa)>=.22: return children,'auto_split'
    return [mask],'split_rejected'


def corollas(filled, auto_split=True):
    n,lab,st,_=cv2.connectedComponentsWithStats(filled,8); out=[]; source=0
    for i in range(1,n):
        area=st[i,cv2.CC_STAT_AREA]*base.MM2_PX
        if not base.AREA_MM2_MIN<=area<=base.AREA_MM2_MAX: continue
        source+=1; mask=(lab==i).astype(np.uint8)
        pieces,status=try_split(mask) if auto_split else ([mask],'disabled')
        for piece_no,p in enumerate(pieces,1):
            m=metrics(p); a=m['area_px']*base.MM2_PX
            if not base.AREA_MM2_MIN<=a<=base.AREA_MM2_MAX or m['aspect']>base.ASPECT_MAX or m['solidity']<base.SOLIDITY_MIN: continue
            mm=cv2.moments(p)
            if not mm['m00']: continue
            out.append(dict(mask=p.astype(bool),source_component_id=source,split_piece=piece_no,
                            split_status=status,cx=mm['m10']/mm['m00'],cy=mm['m01']/mm['m00'],m=m))
    out.sort(key=lambda r:(int(r['cy'])//170,r['cx']))
    return out


def organs(img, corolla_mask, top):
    lc,a,b=base.channels(img); chroma=np.sqrt(a*a+b*b)
    q=((lc<248)&((chroma>2.2)|(b>2)|(a>2.5))&~((lc<115)&(chroma<10))).astype(np.uint8)
    q[:top]=0; q[cv2.dilate(corolla_mask.astype(np.uint8),np.ones((15,15),np.uint8))>0]=0
    q=cv2.morphologyEx(q,cv2.MORPH_OPEN,np.ones((2,2),np.uint8)); merged=np.zeros_like(q)
    for k in (cv2.getStructuringElement(cv2.MORPH_RECT,(13,3)),cv2.getStructuringElement(cv2.MORPH_RECT,(3,13)),
              np.eye(11,dtype=np.uint8),np.fliplr(np.eye(11,dtype=np.uint8))):
        merged |= cv2.morphologyEx(q,cv2.MORPH_CLOSE,k)
    n,lab,st,_=cv2.connectedComponentsWithStats(merged,8); out=[]
    for i in range(1,n):
        area=st[i,cv2.CC_STAT_AREA]*base.MM2_PX
        if not .35<=area<=100: continue
        cs,_=cv2.findContours((lab==i).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not cs: continue
        (cx,cy),(rw,rh),ang=cv2.minAreaRect(max(cs,key=cv2.contourArea)); le,wi=max(rw,rh),min(rw,rh)
        if wi<1: continue
        lm,wm=le*base.MM_PX,wi*base.MM_PX; asp=le/wi
        if 3<=lm<=45 and .15<=wm<=4.5 and asp>=3.2:
            out.append(dict(cx=round(cx,2),cy=round(cy,2),length_mm=round(lm,2),width_mm=round(wm,2),
                            aspect=round(asp,2),angle_deg=round(ang,2),
                            organ_type_auto='unclassified_reproductive_organ',organ_type_FILL='',exclude_FILL=''))
    return sorted(out,key=lambda r:(r['cy'],r['cx']))


def process_sheet(path,folder,out_dir,loc_map=None,auto_split=True):
    island,order=base.ISLANDS.get(folder,(folder,'')); fname=os.path.basename(path); stem=os.path.splitext(fname)[0]
    img=base.load_bgr(path); h,w=img.shape[:2]; loc_map=loc_map or {}; snums,_,_=base.site_numbers(fname)
    isl_sites=sorted({n for isl,n in loc_map if isl==folder}) if loc_map else []
    if len(isl_sites)==1: snums=isl_sites
    if len(snums)==1: site=snums[0]; lat,lon=loc_map.get((folder,site),('','')); candidates=''
    else: site='';lat=lon='';candidates='|'.join(map(str,snums))
    top=specimen_top(img); filled,a,b=foreground_v2(img,top); comps=corollas(filled,auto_split)
    union=np.zeros((h,w),np.uint8)
    for x in comps: union[x['mask']]=1
    oo=organs(img,union,top); brown=(a>6)&((a-b)<-15); ov=img.copy(); rows=[]; centres=[]
    cv2.line(ov,(0,top),(w-1,top),(180,0,180),2)
    for cid,x in enumerate(comps,1):
        comp=x['mask']; spots=base.spot_segment(a,b,comp); area_px=int(comp.sum()); sp=int(spots.sum()); cov=sp/max(area_px,1)
        ns,_,sst,_=cv2.connectedComponentsWithStats(spots,8); nspot=sum(sst[j,cv2.CC_STAT_AREA]*base.MM2_PX>=.02 for j in range(1,ns))
        bf=int((brown&comp).sum())/max(area_px,1); rc,rs=base.orient_base_tip(comp,spots.astype(bool)); g=base.geometry(rc)
        ext=''
        if rs.sum()>10: ext=(rc.shape[0]-1-np.where(rs>0)[0].min())/max(rc.shape[0]-1,1)
        circ=g['throat_width'] if g['n_lobes']>=4 else g['throat_width']*2
        lm=round(g['length']*base.MM_PX,2); wm=round(g['width']*base.MM_PX,2); wl=wm/lm if lm else 0; amm=area_px*base.MM2_PX
        merge='check' if x['split_status'] not in ('not_triggered','auto_split') else ''
        rows.append(dict(island=island,region_order=order,sheet=stem,site_no=site,site_candidates=candidates,
                         site_lat=lat,site_lon=lon,individual_id='',corolla_id=cid,cx=round(x['cx']),cy=round(x['cy']),
                         source_component_id=x['source_component_id'],split_piece=x['split_piece'],split_status=x['split_status'],
                         corolla_len_mm=lm,corolla_width_mm=wm,wl_ratio=round(wl,3),fold_check='check' if wl<.55 else '',
                         merge_check=merge,corolla_area_mm2=round(amm,1),guide_area_mm2=round(sp*base.MM2_PX,2),
                         guide_cov_pct=round(cov*100,2),n_spots=nspot,spot_density_cm2=round(nspot/(amm/100),2),
                         guide_extent_rel=round(ext,3) if ext!='' else '',guide_present=int(cov*100>=.5),
                         brown_frac=round(bf,3),degraded_flag=int(bf>.10),
                         prov_mouth_diam_mm=round(circ/math.pi*base.MM_PX,2),prov_tube_depth_mm=round(g['tube_depth']*base.MM_PX,2),
                         prov_n_lobes=g['n_lobes'],solidity=round(x['m']['solidity'],3),aspect=round(x['m']['aspect'],2),exclude=''))
        centres.append((x['cx'],x['cy'],cid)); cs,_=cv2.findContours(comp.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov,cs,-1,(255,0,255) if x['split_status']=='auto_split' else (0,255,0),3)
        cv2.putText(ov,f'C{cid}',(int(x['cx'])-18,int(x['cy'])),cv2.FONT_HERSHEY_SIMPLEX,1.1,(0,0,255),3)
    for oid,o in enumerate(oo,1):
        xx,yy=int(o['cx']),int(o['cy']); o['nearest_corolla']=min(centres,key=lambda c:(c[0]-xx)**2+(c[1]-yy)**2)[2] if centres else ''
        o.update(island=island,sheet=stem,organ_id=oid); cv2.circle(ov,(xx,yy),8,(0,140,255),-1)
        cv2.putText(ov,f'R{oid} {o["length_mm"]:.1f}mm',(xx+8,yy),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,140,255),2)
    od=Path(out_dir)/'overlays';od.mkdir(parents=True,exist_ok=True);sc=min(1.,1900/max(h,w))
    cv2.imencode('.png',cv2.resize(ov,(int(w*sc),int(h*sc))))[1].tofile(str(od/f'{island}_{stem}.png'))
    return rows,oo


def write_csv(path,rows,fields=None):
    fields=fields or (list(rows[0]) if rows else [])
    with open(path,'w',newline='',encoding='utf-8-sig') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',required=True);ap.add_argument('--out-dir',default='results_v2')
    ap.add_argument('--locations',default='');ap.add_argument('--no-auto-split',action='store_true');a=ap.parse_args();out=Path(a.out_dir);out.mkdir(parents=True,exist_ok=True)
    loc=base.load_locations(a.locations);rr=[];oo=[]
    for folder in sorted(os.listdir(a.data_root)):
        d=Path(a.data_root)/folder
        if not d.is_dir():continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in ('.jpg','.jpeg','.png'):
                r,o=process_sheet(str(p),folder.lower(),str(out),loc,not a.no_auto_split);rr+=r;oo+=o
    if not rr:raise SystemExit('No corollas detected')
    write_csv(out/'traits.csv',rr);of=['island','sheet','organ_id','nearest_corolla','cx','cy','length_mm','width_mm','aspect','angle_deg','organ_type_auto','organ_type_FILL','exclude_FILL']
    write_csv(out/'organs.csv',oo,of);write_csv(out/'styles.csv',oo,of)
    qf=['island','sheet','corolla_id','cx','cy','site_no_auto','site_candidates','site_no_FILL','individual_FILL','flower_no_FILL','fold_state_FILL(open/folded)','split_or_exclude_FILL','site_lat','site_lon','fold_check','merge_check','split_status','notes']
    qr=[]
    for r in rr:qr.append(dict(island=r['island'],sheet=r['sheet'],corolla_id=r['corolla_id'],cx=r['cx'],cy=r['cy'],site_no_auto=r['site_no'],site_candidates=r['site_candidates'],site_no_FILL='',individual_FILL='',flower_no_FILL='',**{'fold_state_FILL(open/folded)':'folded' if r['fold_check'] else '','split_or_exclude_FILL':'CHECK' if r['merge_check'] else ''},site_lat=r['site_lat'],site_lon=r['site_lon'],fold_check=r['fold_check'],merge_check=r['merge_check'],split_status=r['split_status'],notes=''))
    write_csv(out/'qc_plant_labels.csv',qr,qf);print(f'corollas={len(rr)} organs={len(oo)} -> {out}')

if __name__=='__main__':main()
