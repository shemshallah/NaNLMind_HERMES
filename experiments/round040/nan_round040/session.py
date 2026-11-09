#!/usr/bin/env python3
"""Small persistent local-universe CLI. Restricted grammar; exact graph inference.
Usage: python session.py --new; then `add ...`, `ask source target`, `save`, `quit`.
"""
import argparse,json,shlex
from pathlib import Path
from experiment import Native,train_reader,parse_sentence,answer
BASE=Path(__file__).resolve().parent

def run():
 p=argparse.ArgumentParser();p.add_argument('--state',default=str(BASE/'session_state'));p.add_argument('--new',action='store_true');args=p.parse_args()
 state=Path(args.state);state.parent.mkdir(parents=True,exist_ok=True)
 clf=train_reader(17);native=Native();meta=state.with_suffix('.json');blob=state.with_suffix('.bin')
 if not args.new and meta.exists() and blob.exists():
  d=json.loads(meta.read_text());native.records=[tuple(row) for row in d['records']];native.slot=len(native.records)
  assert native.lib.ep_load(native.ptr,str(blob).encode()),'Failed to load native binary'
  print('Restored',len(native.records),'stored facts')
 print('Commands: add <sentence> | ask <source> <target> | save | quit')
 try:
  while True:
   try:s=input('nan://local> ').strip()
   except EOFError:break
   if not s:continue
   cmd,_,rest=s.partition(' ')
   if cmd=='add':
    v=parse_sentence(rest,clf)
    if v is None:print('Unsupported syntax; see README for supported sentence forms.');continue
    a,b,polarity=v;native.append(a,b,polarity,rest);print('Stored:',a,b,polarity)
   elif cmd=='ask':
    xy=rest.split()
    if len(xy)!=2:print('Usage: ask entity1 entity2');continue
    found,path=answer(xy[0],xy[1],native.edges())
    if found:print(f'{xy[0]} is a kind of {xy[1]}. Evidence chain:', ' -> '.join(path))
    else:print('Insufficient evidence; no supported ancestry path found.')
   elif cmd=='save':
    assert native.lib.ep_save(native.ptr,str(blob).encode())
    meta.write_text(json.dumps({'format':'NANEP040','records':native.records},indent=2))
    print('Saved native root and metadata')
   elif cmd in ('quit','exit'):break
   else:print('Unknown command')
 finally:native.close()
if __name__=='__main__':run()
