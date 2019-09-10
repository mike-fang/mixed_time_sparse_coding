let SessionLoad = 1
let s:so_save = &so | let s:siso_save = &siso | set so=0 siso=0
let v:this_session=expand("<sfile>:p")
silent only
cd ~/python_projects/mixed_time_sparse_coding
if expand('%') == '' && !&modified && line('$') <= 1 && getline(1) == ''
  let s:wipebuf = bufnr('%')
endif
set shortmess=aoO
badd +1 ~/python_projects/mixed_time_sparse_coding/hv_lines.py
badd +227 ~/python_projects/mixed_time_sparse_coding/mtsc.py
badd +22 term://.//18503:python\ hv_lines.py
badd +9 ~/python_projects/mixed_time_sparse_coding/hv_line_test.py
badd +71 term://.//19658:python\ hv_lines.py
badd +62 ~/python_projects/mixed_time_sparse_coding/loaders.py
badd +4 ~/python_projects/mixed_time_sparse_coding/scratch_pad.py
badd +10 ~/python_projects/mixed_time_sparse_coding/TODO
badd +19 ~/python_projects/mixed_time_sparse_coding/euler_maruyama.py
badd +26 ~/python_projects/mixed_time_sparse_coding/visualization.py
badd +9 term://.//26713:python\ visualization.py
argglobal
%argdel
$argadd hv_lines.py
edit ~/python_projects/mixed_time_sparse_coding/hv_lines.py
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
exe 'vert 1resize ' . ((&columns * 67 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 10 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 1 + 40) / 80)
argglobal
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
let s:l = 7 - ((2 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
7
normal! 0
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/visualization.py") | buffer ~/python_projects/mixed_time_sparse_coding/visualization.py | else | edit ~/python_projects/mixed_time_sparse_coding/visualization.py | endif
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
13
normal! zo
32
normal! zo
82
normal! zo
141
normal! zo
let s:l = 12 - ((4 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
12
normal! 0
wincmd w
argglobal
if bufexists("~/python_projects/mixed_time_sparse_coding/hv_line_test.py") | buffer ~/python_projects/mixed_time_sparse_coding/hv_line_test.py | else | edit ~/python_projects/mixed_time_sparse_coding/hv_line_test.py | endif
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
let s:l = 93 - ((0 * winheight(0) + 11) / 22)
if s:l < 1 | let s:l = 1 | endif
exe s:l
normal! zt
93
normal! 0
wincmd w
2wincmd w
exe 'vert 1resize ' . ((&columns * 67 + 40) / 80)
exe 'vert 2resize ' . ((&columns * 10 + 40) / 80)
exe 'vert 3resize ' . ((&columns * 1 + 40) / 80)
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
