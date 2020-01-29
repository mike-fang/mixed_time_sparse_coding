let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 ~/python_projects/mixed_time_sparse_coding/vh_patches.py
badd +15 ~/python_projects/mixed_time_sparse_coding/bars_dkl.py
badd +17 ~/python_projects/mixed_time_sparse_coding/vh_mse.py
badd +32 ~/python_projects/mixed_time_sparse_coding/vh_lsc_sweep.py
badd +213 ~/python_projects/mixed_time_sparse_coding/soln_analysis.py
badd +52 ~/python_projects/mixed_time_sparse_coding/vh_dkl.py
badd +4 ~/python_projects/mixed_time_sparse_coding/plt_env.py
argglobal
%argdel
$argadd vh_patches.py
edit ~/python_projects/mixed_time_sparse_coding/vh_lsc_sweep.py
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
exe 'vert 2resize ' . ((&columns * 68 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 67 + 102) / 204)
argglobal
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
20
normal! zo
let s:l = 33 - ((29 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
33
normal! 032|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/vh_dkl.py") | buffer ~/python_projects/mixed_time_sparse_coding/vh_dkl.py | else | edit ~/python_projects/mixed_time_sparse_coding/vh_dkl.py | endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
let s:l = 31 - ((30 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
31
normal! 049|
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/soln_analysis.py") | buffer ~/python_projects/mixed_time_sparse_coding/soln_analysis.py | else | edit ~/python_projects/mixed_time_sparse_coding/soln_analysis.py | endif
setlocal fdm=expr
setlocal fde=SimpylFold#FoldExpr(v:lnum)
setlocal fmr={{{,}}}
setlocal fdi=#
setlocal fdl=0
setlocal fml=1
setlocal fdn=20
setlocal fen
10
normal! zo
118
normal! zo
let s:l = 137 - ((136 * winheight(0) + 26) / 53)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
137
normal! 024|
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 67 + 102) / 204)
exe 'vert 2resize ' . ((&columns * 68 + 102) / 204)
exe 'vert 3resize ' . ((&columns * 67 + 102) / 204)
tabnext 1
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
