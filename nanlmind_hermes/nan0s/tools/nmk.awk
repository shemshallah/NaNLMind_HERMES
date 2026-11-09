# tools/nmk.awk - NaN16 source tool (gawk implementation).
# Outer dev-loop only; guest nasm0 remains the NaN-native assembler. Run under LC_ALL=C.
#   gawk -f tools/nmk.awk -v CMD=emit src.nas > out.nan
#   gawk -f tools/nmk.awk -v CMD=check out.nan
# Word text: NNNN_PPPPPPPP_FFFF with n = NEARidx<<2 | FARidx.
# Guest word bytes: far head hi,lo at 0..1; payload LE at 2..5; near head lo,hi at 6..7.
BEGIN {
  split("7FF1 FFF1 7FF8 FFF8", NE, " ")
  split("F17F F1FF F87F F8FF", FA, " ")
  split("BOOL 0 MOVI 1 MOVHI 2 ADD 3 SUB 4 SHL 5 SHR 6 CMP 7 LD 8 ST 9 JMP 10 CALL 11 RET 12 JMPR 13 NOP 15 VPOKE 16 VPEEK 17 KEY 18 TICK 19 BSWAPM 20 PUTC 21 STW 22 LDN 23 DISPQ 24 DISPSET 25 KEYQ 26 BEEP 27 MUL 32 MULH 33 MULHU 34 DIV 35 MOD 36 DIVU 37 MODU 38 SAR 39 CMPS 40 ADDI 41 LDB 42 STB 43 CALLR 44 BCOPY 45 BFILL 46 FADD 48 FEXP 70 FTANH 71 FLOG 72 FSUB 49 FMUL 50 FDIV 51 FSQRT 52 FCMP 53 CVTIF 54 CVTFI 55 JMPF 56 CALLF 57 FBINFO 58 VSYNC 59 FBCOPY 60 FBFILL 61 OUTB 62 INB 63 INW 64 OUTW 65 OUTSW 66 OUTL 67 CPUID 68 RDTSC 69 HALT 255", OB, " ")
  OSB = "BOOL 0 MOVI 1 MOVHI 2 ADD 3 SUB 4 SHL 5 SHR 6 CMP 7 LD 8 ST 9 JMP 10 CALL 11 RET 12 JMPR 13 NOP 15 VPOKE 16 VPEEK 17 KEY 18 TICK 19 BSWAPM 20 PUTC 21 STW 22 LDN 23 DISPQ 24 DISPSET 25 KEYQ 26 BEEP 27 MUL 32 MULH 33 MULHU 34 DIV 35 MOD 36 DIVU 37 MODU 38 SAR 39 CMPS 40 ADDI 41 LDB 42 STB 43 CALLR 44 BCOPY 45 BFILL 46 FADD 48 FEXP 70 FTANH 71 FLOG 72 FSUB 49 FMUL 50 FDIV 51 FSQRT 52 FCMP 53 CVTIF 54 CVTFI 55 JMPF 56 CALLF 57 FBINFO 58 VSYNC 59 FBCOPY 60 FBFILL 61 OUTB 62 INB 63 INW 64 OUTW 65 OUTSW 66 OUTL 67 CPUID 68 RDTSC 69 HALT 255"
  nob = split(OSB, OB, " ")
  for (i=1; i<nob; i+=2) dn[OB[i]] = OB[i+1]+0
  delete OB
  for (m in dn) opname[dn[m]] = m
  FORMS = "BOOL=dab,ADD=dab,SUB=dab,STW=dab,MOVI=di,MOVHI=di,SHL=dai,SHR=dai,LD=dai,ST=dai,LDN=dai,CMP=ab,JMP=i,CALL=i,RET=,NOP=,HALT=,JMPR=a,KEY=d,DISPQ=d,DISPSET=d,KEYQ=d,BEEP=d,TICK=d,PUTC=d,VPOKE=da,VPEEK=da,BSWAPM=ai,MUL=dab,MULH=dab,MULHU=dab,DIV=dab,MOD=dab,DIVU=dab,MODU=dab,SAR=dai,CMPS=ab,ADDI=dai,LDB=dai,STB=dai,CALLR=a,BCOPY=dab,BFILL=dab,FADD=dab,FEXP=da,FTANH=da,FLOG=da,FSUB=dab,FMUL=dab,FDIV=dab,FSQRT=da,FCMP=ab,CVTIF=da,CVTFI=da,JMPF=j,CALLF=j,FBINFO=d,VSYNC=,FBCOPY=dab,FBFILL=dab,OUTB=da,INB=da,INW=da,OUTW=da,OUTSW=dab,OUTL=da,CPUID=dab,RDTSC=da"
  nf = split(FORMS, FCT, ",")
  for (i=1; i<=nf; i++){ fl=index(FCT[i],"="); FORM[substr(FCT[i],1,fl-1)] = substr(FCT[i],fl+1) }
  delete FCT
  pn["ALWAYS"]=15; pn["NEVER"]=0; pn["EQ"]=4; pn["NE"]=3; pn["LT"]=2; pn["GE"]=5; pn["GT"]=1; pn["LE"]=6
  for (k in pn) prname[pn[k]] = k
  for (i=0;i<256;i++){ CHI[i]=sprintf("%c",i); ORDC[sprintf("%c",i)]=i }
  if (CMD=="") die("usage: -v CMD=emit|check <file>")
  if (CMD=="emit") cmd_emit(ARGV[1])
  else if (CMD=="check") cmd_check(ARGV[1])
  else die("unknown CMD " CMD)
  exit 0
}

function die(m){ print "nmk: " m > "/dev/stderr"; exit 1 }
function strip1(t,k){ return substr(t,k+1) }
function trimx(x){ gsub(/^[ \t]+/,"",x); gsub(/[ \t]+$/,"",x); return x }
function cmtless(raw,   i,n,ch,q,esc){
  # strip ';' comments OUTSIDE double-quoted .str bodies (escapes: \\ and \")
  n=length(raw); q=0; esc=0
  for (i=1;i<=n;i++){
    ch=substr(raw,i,1)
    if (q){
      if (esc) esc=0
      else if (ch=="\\") esc=1
      else if (ch=="\"") q=0
      continue
    }
    if (ch=="\""){ q=1; continue }
    if (ch==";") return substr(raw,1,i-1)
  }
  return raw
}

function inum(x,  s,neg){
  s=x; neg=0
  if (substr(s,1,1)=="-"){ neg=1; s=substr(s,2) }
  else if (substr(s,1,1)=="+") s=substr(s,2)
  if (s ~ /^0[xX][0-9a-fA-F]+$/) return neg ? -hexv(substr(s,3)) : hexv(substr(s,3))
  if (s ~ /^[0-9][0-9]*$/) return neg ? -(s+0) : (s+0)
  die("bad number '" x "'")
}

# expression: terms joined by + -; term = @label | #EQU | number
function ev(x,  s,i,n,ch,sign,term,j,v,tot){
  s=x; gsub(/ /,"",s)
  tot=0; n=length(s); i=0
  while (i<n){
    i++
    ch=substr(s,i,1)
    if (ch=="+" || ch=="-"){ sign=ch; i++; if (i>n) break }
    else sign=""
    j=i
    while (j<=n){
      ch=substr(s,j,1)
      if (ch=="+" || ch=="-") break
      j++
    }
    term=substr(s,i,j-i)
    i=j-1   # leave the next sign unconsumed: the loop's i++ lands on it (r29e fix: '#X-2' used to evaluate as X+2)
    if (term=="") continue
    if (substr(term,1,1)=="@"){
      nm2=substr(term,2)
      if (!(nm2 in LAB)) die("undefined label in expr: " term)
      v=LAB[nm2]
    } else if (substr(term,1,1)=="#"){
      nm2=substr(term,2)
      if (!(nm2 in EQ)) die("undefined equ in expr: " term)
      v=EQ[nm2]
    } else v=inum(term)
    tot += (sign=="-") ? -v : v
  }
  return tot
}

function utf8b(cp){
  if (cp<128) GB[GN++]=cp
  else if (cp<2048){ GB[GN++]=192+int(cp/64); GB[GN++]=128+cp%64 }
  else if (cp<65536){ GB[GN++]=224+int(cp/4096); GB[GN++]=128+int(cp/64)%64; GB[GN++]=128+cp%64 }
  else { GB[GN++]=240+int(cp/262144); GB[GN++]=128+int(cp/4096)%64; GB[GN++]=128+int(cp/64)%64; GB[GN++]=128+cp%64 }
}
function hexv(h,   i,c,v){
  v=0
  for (i=1;i<=length(h);i++){ c=toupper(substr(h,i,1)); v=v*16+index("0123456789ABCDEF",c)-1 }
  return v
}
# estr decodes the .str body into numeric bytes GB[0..GN)
function estr(s,   i,n,c,e){
  GN=0; n=length(s); i=1
  while (i<=n){
    c=substr(s,i,1)
    if (c!="\\"){ GB[GN++]=ordn(c); i++; continue }
    e=substr(s,i+1,1); i+=2
    if (e=="n") GB[GN++]=10
    else if (e=="r") GB[GN++]=13
    else if (e=="t") GB[GN++]=9
    else if (e=="0") GB[GN++]=0
    else if (e=="\\") GB[GN++]=92
    else if (e=="\"") GB[GN++]=34
    else if (e=="'") GB[GN++]=39
    else if (e=="x"){ GB[GN++]=hexv(substr(s,i,2)); i+=2 }
    else if (e=="u"){ utf8b(hexv(substr(s,i,4))); i+=4 }
    else { GB[GN++]=92; GB[GN++]=ordn(e) }
  }
}
# byte representation over the 4-byte window; wrapper quote = double iff contains a quote and no double
function pyre4(base,   out,i,code,ha,hq,q){
  ha=0; hq=0
  for (i=0;i<4;i++){
    code=GB[base+i]
    if (code==39) ha++
    else if (code==34) hq++
  }
  q = (ha>0 && hq==0) ? "\"" : "'"
  out="b" q
  for (i=0;i<4;i++){
    code=GB[base+i]
    if (code==92) out=out "\\\\"
    else if (code==39 && q=="'") out=out "\\'"
    else if (code==34 && q=="'") out=out "\\\""
    else if (code==10) out=out "\\n"
    else if (code==13) out=out "\\r"
    else if (code==9) out=out "\\t"
    else if (code>=32 && code<=126) out=out sprintf("%c",code)
    else out=out sprintf("\\x%02x",code)
  }
  return out q
}
function ordn(c,   k){ k=ORDC[c]; if (k=="") die("non-byte char?"); return k+0 }
function lastq(s,  i){
  for (i=length(s); i>=1; i--) if (substr(s,i,1)=="\"") return i
  return 0
}
function dirname2(p,  d){ d=p; sub(/\/[^\/]*$/,"",d); if (d==p) return "."; return d }

# ---- reader: includes expanded inline (comments stripped), other lines kept verbatim
function readsrc(path,  raw,t,mm2,incn){
  if ((getline raw < path) <= 0) die("cannot read " path)
  do {
    t = trimx(cmtless(raw))
    if (substr(t,1,8)==".include"){
      incn=""
      i2=index(t,"\"")
      if (i2==0) die("bad .include in " path ": " t)
      j2=lastq(t); if (j2<=i2) die("bad .include in " path ": " t)
      incn=substr(t,i2+1,j2-i2-1)
      readsrc(dirname2(path) "/" incn)
    } else {
      SN++; SRCL[SN]=raw
    }
  } while ((getline raw < path) > 0)
  close(path)
}

# ---- pass 1
function parse_item(t,  ii,jj,s1,b1,n4,s4,k4,v4,rest,wi,w,mmla,prd,mm){
  if (t=="") return
  if (t ~ /^[A-Za-z_][A-Za-z0-9_]*:$/){ LAB[substr(t,1,length(t)-1)]=NI; return }
  if (substr(t,1,4)==".equ"){
    split(t, ea, " ")
    if (length(ea)<3) die("bad .equ: " t)
    EQ[ea[2]] = inum(ea[3])
    return
  }
  if (substr(t,1,4)==".org"){
    tg = inum(trimx(strip1(t,5)))
    if (tg < NI) die(".org backwards at " NI)
    while (NI < tg){ IK[NI+1]="data"; IVN[NI+1]=0; IVC[NI+1]="pad"; NI++ }
    return
  }
  if (substr(t,1,4)==".str"){
    ii=index(t,"\""); if (ii==0) die("bad .str: " t)
    jj=lastq(t); if (jj<=ii) die("bad .str: " t)
    s1=substr(t,ii+1,jj-ii-1)
    estr(s1)
    GB[GN++]=0
    while (GN%4 != 0) GB[GN++]=0
    n4=GN/4
    for (s4=0; s4<n4; s4++){
      v4=GB[s4*4] + GB[s4*4+1]*256 + GB[s4*4+2]*65536 + GB[s4*4+3]*16777216
      IK[NI+1]="data"; IVN[NI+1]=v4; IVC[NI+1]="str " pyre4(s4*4); NI++
    }
    return
  }
  if (substr(t,1,5)==".word"){
    rest = strip1(t,5)
    split(rest, wa, ",")
    for (wi=1; wi<=length(wa); wi++){
      w = trimx(wa[wi])
      if (w != ""){ IK[NI+1]="data"; IVN[NI+1]=-2; IVE[NI+1]=w; IVC[NI+1]="word"; NI++ }
    }
    return
  }
  if (t ~ /^LA(\[[A-Za-z_][A-Za-z0-9_]*\])?[ \t]+r[0-9]+[ \t]*,/){
    lr=substr(t,3); lp=""
    if (substr(lr,1,1)=="["){ le=index(lr,"]"); lp=substr(lr,2,le-2); lr=substr(lr,le+1) }
    sub(/^[ \t]+r/,"",lr); match(lr,/^[0-9]+/); lrn=substr(lr,1,RLENGTH)+0; lr=substr(lr,RLENGTH+1); sub(/^[ \t]*,[ \t]*/,"",lr)
    prd = (lp=="") ? 15 : pn[lp]
    IK[NI+1]="la"; IR[NI+1]=lrn; IX[NI+1]=trimx(lr); IH[NI+1]=0; IP_[NI+1]=prd; NI++
    IK[NI+1]="la"; IR[NI+1]=lrn; IX[NI+1]=trimx(lr); IH[NI+1]=1; IP_[NI+1]=prd; NI++
    return
  }
  IK[NI+1]="ins"; IT[NI+1]=t; NI++
}

# ---- pass 2
function regnum(x){
  if (x !~ /^[rR][0-9][0-9]?$/) die("expected register, got '" x "'")
  rn2=substr(x,2)+0
  if (rn2>15) die("register out of range: " x)
  return rn2
}
function wline(n,p){
  p=p%4294967296; if (p<0) p+=4294967296
  return sprintf("%s_%08X_%s", NE[int(n/4)+1], p, FA[(n%4)+1])
}
function dis2(n,p,  op,d,imm,nm,pr,f,ci,ch,x2,sp2,b2){
  p=p%4294967296; if (p<0) p+=4294967296
  op=int(p/16777216); d=int(p/1048576)%16; a2=int(p/65536)%16; imm=p%65536
  nm=opname[op]
  if (nm=="BOOL") return "BOOL." n " r" d ",r" a2 ",r" (imm%16)
  if (n==15) pr=""
  else if (n in prname) pr="[" prname[n] "]"
  else pr="[n" n "]"
  f=FORM[nm]
  b2="r" (imm%16)
  if (nm=="CMP") sp2="r" a2 ",r" (imm%16)
  else if (nm=="STW"||nm=="MUL"||nm=="MULH"||nm=="MULHU"||nm=="DIV"||nm=="MOD"||nm=="DIVU"||nm=="MODU"||nm=="BCOPY"||nm=="BFILL"||nm=="FADD"||nm=="FSUB"||nm=="FMUL"||nm=="FDIV") sp2="r" d ",r" a2 ",r" (imm%16)
  else {
    sp2=""
    for (ci=1;ci<=length(f);ci++){
      ch=substr(f,ci,1)
      if (ch=="d") x2="r" d
      else if (ch=="a") x2="r" a2
      else if (ch=="b") x2=b2
      else if (ch=="i") x2="" imm
      else if (ch=="j") x2="" (p%16777216)
      else x2=""
      sp2 = (sp2=="") ? x2 : sp2 "," x2
    }
  }
  return nm pr (sp2=="" ? "" : " " sp2)
}

function resolve(   i,t,v,mn,bn,predn,rest,args,na,ci,ch,x,f,d,a,imm0,imm2,p,tj2,rn2,mmp){
  for (j=1;j<=NI;j++){
    if (IK[j]=="la"){
      v=ev(IX[j]); v=v%4294967296; if (v<0) v+=4294967296
      if (IH[j]==0) p = dn["MOVI"]*16777216 + IR[j]*1048576 + (v%65536)
      else p = dn["MOVHI"]*16777216 + IR[j]*1048576 + int(v/65536)
      EMIT[j-1]=wline(IP_[j],p)"   ; "sprintf("%03d",j-1)" "dis2(IP_[j],p)
      continue
    }
    if (IK[j]=="data"){
      if (IVN[j]==-2) v=ev(IVE[j]); else v=IVN[j]
      v=v%4294967296; if (v<0) v+=4294967296
      EMIT[j-1]=wline(10,v)"   ; "sprintf("%03d",j-1)" "IVC[j]
      continue
    }
    t=IT[j]
    t=IT[j]
    if (t=="" || !match(t, /^[A-Za-z_][A-Za-z0-9_]*/)) die("bad line [" i ",len" length(t) "]: [" t "]")
    mn=substr(t,1,RLENGTH); mr=substr(t,RLENGTH+1); bn=""; predn=""
    if (match(mr,/^\.[0-9]+/)){ bn=substr(mr,2,RLENGTH-1); mr=substr(mr,RLENGTH+1) }
    else if (match(mr,/^\[[A-Za-z_][A-Za-z0-9_]*\]/)){ predn=substr(mr,2,RLENGTH-2); mr=substr(mr,RLENGTH+1) }
    rest=trimx(mr)
    if (bn!="") n=bn+0
    else if (predn!="") n=pn[predn]
    else n=15
    if (!(mn in FORM)) die("unknown mnemonic: " mn)
    f=FORM[mn]
    na=0
    if (rest!=""){ na=split(rest,args,","); for (ci=1;ci<=na;ci++) args[ci]=trimx(args[ci]) }
    d=0; a=0; imm0=0
    lim=(na<length(f)) ? na : length(f)
    for (ci=1; ci<=lim; ci++){
      ch=substr(f,ci,1); x=args[ci]
      if (ch=="d") d=regnum(x)
      else if (ch=="a") a=regnum(x)
      else if (ch=="b") imm0=regnum(x)
      else if (ch=="j"){ tj2=ev(x); tj2=tj2%16777216; if (tj2<0) tj2+=16777216; d=int(tj2/1048576); a=int(tj2/65536)%16; imm0=tj2%65536 }
      else imm0=ev(x)
    }
    if (mn=="MOVI" && imm0>65535) die(sprintf("MOVI immediate %d does not fit 16 bits (use LA): %s", imm0, t))
    imm2=imm0%65536; if (imm2<0) imm2+=65536
    p = dn[mn]*16777216 + d*1048576 + a*65536 + imm2
    EMIT[j-1]=wline(n,p)"   ; "sprintf("%03d",j-1)" "dis2(n,p)
  }
}

function cmd_emit(src,   i){
  readsrc(src)
  for (i=1;i<=SN;i++) parse_item(trimx(cmtless(SRCL[i])))
  resolve()
  print "; NaN16 guest program, generated by scaffold from mnemonics; word index in comments"
  for (i=0;i<NI;i++) print EMIT[i]
  if (LABELS != "") { for (k in LAB) printf "%s %d\n", k, LAB[k] > LABELS; close(LABELS) }   # -v LABELS=file: "name word-index" lines (unsorted)
  printf "words %d\n", NI > "/dev/stderr"
}

# ---- check: compare dis2 decode against the comment column
function nrev(tab,h,  i){ for (i=1;i<=4;i++) if (tab[i]==h) return i-1; die("bad head " h) }
function cmd_check(f,  l,ii,hx,cmt,hn,pp,ff,n2,p2,i2,tx2,kk,nbad){
  nbad=0
  while ((getline l < f) > 0){
    if (substr(l,1,1)==";" || l=="") continue
    ii=index(l,";"); if (ii==0) continue
    hx=trimx(substr(l,1,ii-1)); cmt=trimx(substr(l,ii+1))
    if (hx=="") continue
    if (hx !~ /^[0-9A-Fa-f_]+$/) continue   # labels/.equ/.org lines: not checkable code rows
    split(hx, hh, "_")
    n2=nrev(NE,hh[1])*4 + nrev(FA,hh[3])
    pp=hh[2]; p2=hexv(pp)
    sp2=index(cmt," ")
    if (sp2>0){ i2=substr(cmt,1,sp2-1); tx2=trimx(substr(cmt,sp2+1)) }
    else { i2=cmt; tx2="" }
    if (substr(tx2,1,3)=="str" || tx2=="word" || tx2=="pad") continue
    kk=dis2(n2,p2); gsub(/[ \t]/,"",kk); gsub(/[ \t]/,"",tx2)
    if (kk!=tx2){ nbad++; print "MISMATCH " i2 ": " kk " <> " tx2 > "/dev/stderr" }
  }
  close(f)
  print "check: " nbad " mismatches" > "/dev/stderr"
  if (nbad>0) exit 1
}
