/* ---------- real water physics + fishing: everything below is simulated live, nothing is a looping clip ---------- */
/* Water: a GPU wave-equation solver (anisotropic for perspective, damped, with a shore mask) feeds refraction + glints in the background shader.
   Fishing: verlet fishing line, spring-damper float, ballistic droplets that land back on the surface and spawn their own rings. */
const WATER={
  hero1:[[.40,.60],[.52,.60],[.62,.66],[.80,.78],[.92,.9],[1,.93],[1,1],[0,1],[0,.88],[.10,.82],[.25,.73],[.36,.65]],
  hero2:[[.42,.60],[.56,.60],[.70,.68],[.88,.77],[1,.82],[1,1],[0,1],[0,.9],[.12,.84],[.28,.74],[.38,.65]],
  pond1:[[0,.66],[.62,.66],[.68,.69],[.68,.74],[.62,.80],[.62,.84],[.68,.92],[.72,1],[0,1]],
  pond2:[[0,.61],[.45,.61],[.62,.63],[.61,.70],[.68,.76],[.72,.84],[.78,1],[0,1]]};
const WP={hero1:[1536,495],hero2:[1536,526],pond1:[1600,667],pond2:[1600,848]};
const FIX={tip:[1135/1600,91/667],flt:[783/1600,515/667],base:[.456,.668]};
const SIMW=512,SIMH=256;
const sim={ok:false,img:'',cur:0,tex:[],fbo:[]},masks={},fx={ok:false};
const imps=[],impBuf=new Float32Array(48),drops=[];
let lnbuf,lnvao;const lnBuf=new Float32Array(8*260);let lnN=0,flDraw=null;
const inPoly=(poly,x,y)=>{let c=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])c=!c}return c};
const addImp=(img,u,v,r,s)=>{if(sim.ok&&sim.img===img&&imps.length<96)imps.push([u,v,r,s])};
const inv=(L,s)=>{const c=L.c0,f=L.c1;return[f[0]+((s[0]-.5)*c[2]+c[0]-f[0])/f[2],f[1]+((s[1]-.5)*c[3]+c[1]-f[1])/f[2]]};

const SIMFS=`#version 300 es
precision highp float;in vec2 v;out vec4 o;
uniform sampler2D sT,mT;uniform vec2 sz;uniform float wy,ar,ar2,damp;uniform vec4 imp[12];uniform int nimp;
void main(){
  vec2 uv=v,d=1./sz;vec2 c=texture(sT,uv).rg;float h=c.x,ho=c.y;
  float hl=texture(sT,uv-vec2(d.x,0.)).x,hr=texture(sT,uv+vec2(d.x,0.)).x,hu=texture(sT,uv-vec2(0.,d.y)).x,hd=texture(sT,uv+vec2(0.,d.y)).x;
  float t=clamp((uv.y-wy)/(1.-wy),0.,1.);float f=mix(.2,.62,t);float A=f*f*ar2;float C=min(.9/(1.+A),.46);
  float m=texture(mT,uv).r;
  float lap=(hl+hr-2.*h)+A*(hu+hd-2.*h);
  float nh=2.*h-ho+C*lap;
  nh+=.018*(.25*(hl+hr+hu+hd)-h);                       // viscosity: short capillary ripples die first
  nh*=mix(.84,damp,m);                                   // the shore absorbs
  vec2 e=min(uv,1.-uv);nh*=mix(.88,1.,smoothstep(0.,.025,min(e.x,e.y)));
  for(int i=0;i<12;i++){if(i>=nimp)break;vec4 q=imp[i];float fq=mix(.2,.62,clamp((q.y-wy)/(1.-wy),0.,1.));
    vec2 dd=vec2((uv.x-q.x)*ar,(uv.y-q.y)/fq);nh+=q.w*exp(-dot(dd,dd)/(q.z*q.z))*m;}
  o=vec4(nh,h,0.,1.);}`;
const LNVS=`#version 300 es
layout(location=0)in vec4 sg;layout(location=1)in vec4 st;uniform vec2 res;out float dd;out float ww;out float al;out float tn;
void main(){vec2 p0=sg.xy,p1=sg.zw;vec2 dv=p1-p0;float L=max(length(dv),.001);vec2 tg=dv/L;vec2 nm=vec2(-tg.y,tg.x);
  float k=float(gl_VertexID&1),s=float(gl_VertexID>>1)*2.-1.;float hw=st.x*.5+1.;
  vec2 pos=mix(p0,p1,k)+tg*(k*2.-1.)*hw*.5+nm*s*hw;dd=s*hw;ww=st.x*.5;al=st.y;tn=st.z;
  gl_Position=vec4(pos.x/res.x*2.-1.,1.-pos.y/res.y*2.,0,1);}`;
const LNFS=`#version 300 es
precision highp float;in float dd;in float ww;in float al;in float tn;out vec4 o;
void main(){float a=1.-smoothstep(ww-.5,ww+.5,abs(dd));vec3 c=mix(vec3(.80,.86,.92),vec3(1.,.95,.84),tn);o=vec4(c,a*al);}`;
const FLVS=`#version 300 es
uniform vec4 fb;uniform vec2 res;out vec2 q;
void main(){vec2 k=vec2(gl_VertexID&1,gl_VertexID>>1)*2.-1.;q=k;vec2 pos=fb.xy+k*fb.zw;gl_Position=vec4(pos.x/res.x*2.-1.,1.-pos.y/res.y*2.,0,1);}`;
const FLFS=`#version 300 es
precision highp float;in vec2 q;out vec4 o;uniform float sink,alpha,air;
void main(){
  float wl=mix(.45,3.,air);float cy=-.35+sink*1.55;
  float body=length(vec2(q.x/.42,(q.y-cy)/.78))-1.;
  float above=step(q.y,wl);float aa=1.-smoothstep(-.07,.05,body);
  vec3 red=vec3(.9,.26,.17),ivory=vec3(.95,.91,.8);
  float sh=.55+.45*sqrt(max(1.-(q.x/.42)*(q.x/.42),0.));
  vec3 c=(q.y-cy)<-.36?red:ivory;c*=sh;
  c+=vec3(.55,.5,.4)*pow(max(1.-length(vec2((q.x+.15)/.2,(q.y-cy+.45)/.3)),0.),2.);
  float underA=aa*(1.-above)*.34*smoothstep(1.2,.2,q.y-wl);
  float men=exp(-pow((q.y-wl)/.06,2.))*exp(-pow(q.x/.85,2.))*.4*(1.-air);
  vec3 col=c*(above>.5?1.:.5)+vec3(.7,.8,.92)*men;
  float a=(aa*above+underA+men)*alpha;o=vec4(col,clamp(a,0.,1.));}`;

function mkMask(poly){
  const c=document.createElement('canvas');c.width=256;c.height=128;const x=c.getContext('2d');x.fillStyle='#000';x.fillRect(0,0,256,128);
  try{x.filter='blur(2.2px)'}catch(e){}
  x.fillStyle='#fff';x.beginPath();
  poly.forEach((q,i)=>{const X=(q[0]<=0?-.08:q[0]>=1?1.08:q[0])*256,Y=(q[1]>=1?1.1:q[1])*128;i?x.lineTo(X,Y):x.moveTo(X,Y)});x.closePath();x.fill();
  const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,c);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t}

function initPhysics(){
  progs.sim=mk(FSVS,SIMFS);progs.ln=mk(LNVS,LNFS);progs.fl=mk(FLVS,FLFS);
  lnbuf=gl.createBuffer();lnvao=gl.createVertexArray();gl.bindVertexArray(lnvao);gl.bindBuffer(gl.ARRAY_BUFFER,lnbuf);
  for(const[l,o]of[[0,0],[1,16]]){gl.enableVertexAttribArray(l);gl.vertexAttribPointer(l,4,gl.FLOAT,false,32,o);gl.vertexAttribDivisor(l,1)}
  gl.bindVertexArray(null);fx.ok=true;
  for(const k in WATER)masks[k]=mkMask(WATER[k]);
  const none=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,none);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,0,0,255]));
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);masks.none=none;
  if(gl.getExtension('EXT_color_buffer_float')||gl.getExtension('EXT_color_buffer_half_float')){
    for(let i=0;i<2;i++){const t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA16F,SIMW,SIMH,0,gl.RGBA,gl.HALF_FLOAT,null);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
      const f=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,t,0);
      if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error('wave solver framebuffer unsupported');
      sim.tex.push(t);sim.fbo.push(f)}
    sim.ok=!reduce;simClear();
  }
}
function simClear(){for(const f of sim.fbo){gl.bindFramebuffer(gl.FRAMEBUFFER,f);gl.clearColor(0,0,0,0);gl.clear(gl.COLOR_BUFFER_BIT)}imps.length=0}
function simStep(dt){
  const g=progs.sim;gl.useProgram(g);gl.bindVertexArray(null);gl.disable(gl.BLEND);gl.viewport(0,0,SIMW,SIMH);
  const im=IM[sim.img];
  gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,masks[sim.img]);gl.uniform1i(g.u('mT'),3);gl.uniform1i(g.u('sT'),2);
  gl.uniform2f(g.u('sz'),SIMW,SIMH);gl.uniform1f(g.u('wy'),im.wy);gl.uniform1f(g.u('ar'),im.ar);gl.uniform1f(g.u('ar2'),Math.pow(im.ar*SIMH/SIMW,2));gl.uniform1f(g.u('damp'),.9974);
  const steps=clamp(Math.round(dt*140),1,4);
  for(let s=0;s<steps;s++){
    const b=imps.splice(0,12);impBuf.fill(0);b.forEach((q,i)=>impBuf.set(q,i*4));gl.uniform4fv(g.u('imp'),impBuf);gl.uniform1i(g.u('nimp'),b.length);
    gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,sim.tex[sim.cur]);
    gl.bindFramebuffer(gl.FRAMEBUFFER,sim.fbo[1-sim.cur]);gl.drawArrays(gl.TRIANGLES,0,3);sim.cur=1-sim.cur;
  }
}

/* ----- fishing line: verlet rope, px space of the pond1 image ----- */
const RN=26,rope={x:new Float32Array(RN),y:new Float32Array(RN),ox:new Float32Array(RN),oy:new Float32Array(RN),acc:0,ready:false};
const fl={dy:0,vy:0,ringT:1,dragT:0};
function ropeReset(a,b){for(let i=0;i<RN;i++){const t=i/(RN-1);rope.x[i]=rope.ox[i]=a[0]+(b[0]-a[0])*t;rope.y[i]=rope.oy[i]=a[1]+(b[1]-a[1])*t}rope.ready=true}
function ropeSim(dt,A,B,slack,tm){
  const chord=Math.hypot(B[0]-A[0],B[1]-A[1]),seg=chord*(1+slack)/(RN-1);const h=1/120;rope.acc=Math.min(rope.acc+dt,.05);
  while(rope.acc>=h){rope.acc-=h;
    for(let i=1;i<RN-1;i++){const vx=(rope.x[i]-rope.ox[i])*.991,vy=(rope.y[i]-rope.oy[i])*.991;rope.ox[i]=rope.x[i];rope.oy[i]=rope.y[i];
      rope.x[i]+=vx+Math.sin(tm*.8+i*.45)*.003;rope.y[i]+=vy+2300*h*h}
    rope.x[0]=A[0];rope.y[0]=A[1];rope.x[RN-1]=B[0];rope.y[RN-1]=B[1];
    for(let it=0;it<20;it++){
      for(let i=0;i<RN-1;i++){const dx=rope.x[i+1]-rope.x[i],dy=rope.y[i+1]-rope.y[i],d=Math.hypot(dx,dy)||1e-4,df=(d-seg)/d;let a0=.5,a1=.5;if(i===0){a0=0;a1=1}else if(i===RN-2){a0=1;a1=0}
        rope.x[i]+=dx*df*a0;rope.y[i]+=dy*df*a0;rope.x[i+1]-=dx*df*a1;rope.y[i+1]-=dy*df*a1}
      rope.x[0]=A[0];rope.y[0]=A[1];rope.x[RN-1]=B[0];rope.y[RN-1]=B[1]}}}

/* ----- droplets + events ----- */
const splash=(img,u,v,n,vmin,vmax,spread,str,r)=>{const[w,h]=WP[img];addImp(img,u,v,r||.014,str);
  for(let i=0;i<n&&drops.length<150;i++){const a=-Math.PI/2+(Math.random()-.5)*spread,s=vmin+Math.random()*(vmax-vmin);
    drops.push({img,x:u*w+(Math.random()-.5)*12,y:v*h-2,vx:Math.cos(a)*s*.75,vy:Math.sin(a)*s,wl:v*h+(Math.random()-.4)*14})}};
const EVENTS=[
  {p:.213,fn:()=>splash('pond1',FIX.flt[0],FIX.flt[1],18,240,600,1.2,1.0,.013)},          // the cast lands
  {p:.262,fn:()=>{fl.vy+=150;addImp('pond1',FIX.flt[0],FIX.flt[1],.006,.35)}},             // first nibble
  {p:.288,fn:()=>{fl.vy+=230;splash('pond1',FIX.flt[0],FIX.flt[1],6,120,300,1,.5,.008)}},  // second nibble
  {p:.312,fn:()=>{fl.vy+=480;splash('pond1',FIX.flt[0],FIX.flt[1],10,160,420,1.1,.85,.011)}}, // the hook sets
  {p:.405,fn:()=>splash('pond2',FIX.base[0],FIX.base[1],46,260,780,1.7,1.25,.022)},        // the carp breaks the surface
  {p:.438,fn:()=>addImp('pond2',FIX.base[0]-.02,FIX.base[1]+.01,.012,.55)},                // tail slap
  {p:.462,fn:()=>splash('pond2',FIX.base[0]-.012,FIX.base[1]+.004,26,200,520,1.4,.9,.018)} // it falls back
];
function updateDrops(dt){
  for(let i=drops.length-1;i>=0;i--){const d=drops[i];d.vy+=2600*dt;d.x+=d.vx*dt;d.y+=d.vy*dt;
    if(d.vy>0&&d.y>=d.wl){const[w,h]=WP[d.img];addImp(d.img,d.x/w,d.y/h,.0034,.06+Math.random()*.06);drops.splice(i,1)}}}

const putLn=(x0,y0,x1,y1,w,a,t)=>{if(a<.01||lnN>=260)return;const o=lnN*8;lnBuf[o]=x0;lnBuf[o+1]=y0;lnBuf[o+2]=x1;lnBuf[o+3]=y1;lnBuf[o+4]=w;lnBuf[o+5]=a;lnBuf[o+6]=t;lnBuf[o+7]=0;lnN++};
const simFlag={a:0,b:0};
let pPrev=0,lpx=-9,lpy=-9,wakeT=0,wakeS=1,ambT=0,boatT=0,rainT=0,clickAt=null;
addEventListener('pointerdown',e=>{clickAt=[e.clientX/innerWidth,e.clientY/innerHeight]});

function physics(sc,p,tm,dt,wb,dpr){
  lnN=0;flDraw=null;
  const simImg=(sc.B&&sc.t>.5)?sc.B.n:sc.A.n,L=(n)=>sc.A.n===n?sc.A:(sc.B&&sc.B.n===n?sc.B:null),vis=(l)=>l===sc.A?1-wb:wb;
  const onWater=!!WATER[simImg]&&masks[simImg];
  if(sim.ok&&simImg!==sim.img){sim.img=simImg;simClear()}
  simFlag.a=sim.ok&&onWater&&sc.A.n===simImg?1:0;simFlag.b=sim.ok&&onWater&&sc.B&&sc.B.n===simImg?1:0;
  const jump=Math.abs(p-pPrev)>.02,fire=!jump&&!reduce,Ls=L(simImg);
  if(fire)for(const ev of EVENTS)if(pPrev<ev.p&&p>=ev.p)ev.fn();
  if(fx.ok&&!reduce)updateDrops(dt);else drops.length=0;
  if(sim.ok&&onWater&&Ls&&!reduce){
    // snow settling on the water + occasional raindrop-sized disturbances
    ambT-=dt;if(ambT<=0){ambT=(.45+Math.random())/(simImg.startsWith('hero')?6:3.4);const wy=IM[simImg].wy;
      for(let k=0;k<8;k++){const u=Math.random(),v=wy+(1-wy)*Math.pow(Math.random(),.65);if(inPoly(WATER[simImg],u,v)){addImp(simImg,u,v,.0032+Math.random()*.004,.05+Math.random()*.1);break}}}
    // pointer wake
    if(px>-5){const dxs=(px-lpx)*innerWidth*.5,dys=(py-lpy)*innerHeight*.5,spd=Math.hypot(dxs,dys);
      if(lpx>-5&&spd>3){const q=inv(Ls,[(px+1)/2,(py+1)/2]);if(inPoly(WATER[simImg],q[0],q[1])){wakeT-=dt;if(wakeT<=0){wakeT=.035;wakeS=-wakeS;const k=Math.min(1,spd/40);addImp(simImg,q[0],q[1],.004+.003*k,wakeS*(.05+.05*k))}}}
      lpx=px;lpy=py}
    // a tap / click drops a stone
    if(clickAt){const q=inv(Ls,clickAt);if(inPoly(WATER[simImg],q[0],q[1])){addImp(simImg,q[0],q[1],.011,.55);if(WP[simImg])splash(simImg,q[0],q[1],7,120,320,1.3,.0,.008)}clickAt=null}
    // the camera is a boat: scrolling pushes a bow wave
    if(simImg.startsWith('hero')&&vel>.16){boatT-=dt;if(boatT<=0){boatT=.06;for(const sx of[-.07,.07]){const q=inv(Ls,[.5+sx,.93]);addImp(simImg,q[0],q[1],.007,.07*vel)}}}
    // droplets falling out of the carp's spray
    if(simImg==='pond2'&&p>.41&&p<.5){rainT-=dt;if(rainT<=0){rainT=.045;const[w,h]=WP.pond2;drops.push({img:'pond2',x:(.40+Math.random()*.11)*w,y:(.55+Math.random()*.06)*h,vx:(Math.random()-.5)*40,vy:Math.random()*60,wl:(.655+Math.random()*.035)*h})}}
  }else clickAt=null;
  if(sim.ok&&onWater)simStep(dt);

  // ---- the line, the float, the droplets ----
  if(fx.ok){
    const L1=L('pond1'),[w1,h1]=WP.pond1;
    if(L1&&p>.15&&p<.43){
      const tip=[FIX.tip[0]*w1,FIX.tip[1]*h1],F=[FIX.flt[0]*w1,FIX.flt[1]*h1],wlY=F[1]+2;
      if(!rope.ready||jump||p<.16){ropeReset(tip,[F[0],wlY-22]);fl.dy=0;fl.vy=0}
      const e=clamp((p-.17)/.045),ee=e*e*(3-2*e),flying=e<1;
      const hook=sm(.312,.33,p),drag=sm(.318,.36,p),taut=hook*(1-sm(.36,.4,p));
      fl.vy+=(-55*fl.dy-3.4*fl.vy)*dt;fl.dy=clamp(fl.dy+fl.vy*dt,-6,40);
      const bob=Math.sin(tm*1.9)*.9+Math.sin(tm*3.3+1)*.4;
      const sub=clamp((fl.dy+bob)/16,0,1)*(1-hook)+hook;
      const fx_=F[0]-drag*46+Math.sin(tm*.7)*.5;
      let B;
      if(flying){B=[lerp(tip[0],fx_,ee),lerp(tip[1],wlY-22,ee)-Math.sin(Math.PI*ee)*.08*h1]}
      else B=[fx_,wlY-22*(1-sub)];
      ropeSim(dt,tip,B,flying?.03:lerp(.018,.003,hook),tm);
      // rings that belong to the float
      if(!flying&&!reduce){fl.ringT-=dt;if(fl.ringT<=0){fl.ringT=1.7+Math.random()*1.4;addImp('pond1',fx_/w1,wlY/h1,.005,.08+Math.min(.2,Math.abs(fl.vy)*.002))}
        if(drag>.02){fl.dragT-=dt;if(fl.dragT<=0){fl.dragT=.07;wakeS=-wakeS;addImp('pond1',fx_/w1,wlY/h1,.007,wakeS*.13)}}}
      const a=vis(L1)*sm(.165,.18,p);
      const sx=L1.c1[2]/L1.c0[2]*W/w1,sy=L1.c1[2]/L1.c0[3]*H/h1;
      const S=(x,y)=>{const s_=scr(L1,[x/w1,y/h1]);return[s_[0]*W,s_[1]*H]};
      let prev=null;
      for(let i=0;i<RN;i++){const k=Math.sin(Math.PI*i/(RN-1)),tx=rope.x[i]+taut*1.5*Math.sin(tm*41+i*1.2)*k;const cur=S(tx,rope.y[i]);if(prev)putLn(prev[0],prev[1],cur[0],cur[1],1.15*dpr,(.5+.3*taut)*a,.15+.5*taut);prev=cur}
      const c=S(B[0],flying?B[1]+20:wlY);flDraw={air:flying?1:0,x:c[0],y:c[1],hw:6*sx,hh:14*sy,sink:flying?0:clamp(sub,0,1.02),alpha:a*(sub>.995&&hook>.99?.0:1)}
    }
    // droplets: little streaks along their velocity
    for(const d of drops){const Ld=L(d.img);if(!Ld)continue;const[w,h]=WP[d.img],s0=scr(Ld,[d.x/w,d.y/h]),s1=scr(Ld,[(d.x-d.vx*.014)/w,(d.y-d.vy*.014)/h]);
      putLn(s0[0]*W,s0[1]*H,s1[0]*W,s1[1]*H,1.4*dpr,.8*vis(Ld),1)}
  }
  pPrev=p;
}
function drawFx(){
  if(!fx.ok)return;let g;
  gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  if(lnN>0){g=progs.ln;gl.useProgram(g);gl.bindVertexArray(lnvao);gl.bindBuffer(gl.ARRAY_BUFFER,lnbuf);gl.bufferData(gl.ARRAY_BUFFER,lnBuf.subarray(0,lnN*8),gl.DYNAMIC_DRAW);
    gl.uniform2f(g.u('res'),W,H);gl.drawArraysInstanced(gl.TRIANGLE_STRIP,0,4,lnN)}
  if(flDraw&&flDraw.alpha>.01){g=progs.fl;gl.useProgram(g);gl.bindVertexArray(null);
    gl.uniform4f(g.u('fb'),flDraw.x,flDraw.y-.45*flDraw.hh,flDraw.hw,flDraw.hh);gl.uniform2f(g.u('res'),W,H);gl.uniform1f(g.u('sink'),flDraw.sink);gl.uniform1f(g.u('alpha'),flDraw.alpha);gl.uniform1f(g.u('air'),flDraw.air);gl.drawArrays(gl.TRIANGLE_STRIP,0,4)}
  gl.disable(gl.BLEND);gl.bindVertexArray(null);
}
dbg.phys=()=>({sim:sim.ok,fx:fx.ok,img:sim.img,drops:drops.length,imps:imps.length,fl:{dy:+fl.dy.toFixed(1),vy:+fl.vy.toFixed(1)},ln:lnN,rope:[rope.x[RN>>1],rope.y[RN>>1]],glerr:gl.getError()});
