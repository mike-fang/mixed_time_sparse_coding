let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +16 ~/python_projects/mixed_time_sparse_coding/vh_learn_pi.py
badd +37 ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py
badd +0 ~/python_projects/mixed_time_sparse_coding/loaders.py
badd +0 term://~/python_projects/mixed_time_sparse_coding//8773:python\ loaders.py
argglobal
%argdel
$argadd vh_learn_pi.py
set stal=2
edit ~/python_projects/mixed_time_sparse_coding/vh_learn_pi.py
set splitbelow splitright
wincmd _ | wincmd |
vsplit
wincmd _ | wincmd |
vsplit
2wincmd h
wincmd w
wincmd w
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
exe 'vert 1resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 68 + 102) / 204)
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 16 - ((15 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
16
normal! 09|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/vh_no_norm.py") | buffer ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py | else | edit ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/vh_no_norm.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
9
normal! zo
let s:l = 12 - ((11 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 048|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/loaders.py") | buffer ~/python_projects/mixed_time_sparse_coding/loaders.py | else | edit ~/python_projects/mixed_time_sparse_coding/loaders.py | endif
if &buftype ==# 'terminal'
  silent file ~/python_projects/mixed_time_sparse_coding/loaders.py
endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
1
normal! zo
335
normal! zo
433
normal! zo
let s:l = 443 - ((442 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
443
normal! 038|
wincmd w
exe 'vert 1resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 68 + 102) / 204)
tabnew
set splitbelow splitright
wincmd t
set winminheight=0
set winheight=1
set winminwidth=0
set winwidth=1
argglobal
if bufexists("term://~/python_projects/mixed_time_sparse_coding//8773:python\ loaders.py") | buffer term://~/python_projects/mixed_time_sparse_coding//8773:python\ loaders.py | else | edit term://~/python_projects/mixed_time_sparse_coding//8773:python\ loaders.py | endif
if &buftype ==# 'terminal'
  silent file term://~/python_projects/mixed_time_sparse_coding//8773:python\ loaders.py
endif
setlocal fdm=manual
setlocal fde=0
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 2 - ((1 * winheight(0) + 26) / 52)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
2
normal! 018|
tabnext 2
set stal=1
if exists('s:wipebuf') && getbufvar(s:wipebuf, '&buftype') isnot# 'terminal'
  silent exe 'bwipe ' . s:wipebuf
endif
unlet! s:wipebuf
set winheight=1 winwidth=20 winminheight=1 winminwidth=1 shortmess=filnxtToOF
let s:sx = expand("<sfile>:p:r")."x.vim"
if file_readable(s:sx)
  exe "source " . fnameescape(s:sx)
endif
let &so = s:so_save | let &siso = s:siso_save
doautoall SessionLoadPost
unlet SessionLoad
" vim: set ft=vim :
