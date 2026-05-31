(function(){
  "use strict";
  // Result tabs
  var tabs=document.querySelectorAll("#resultTabs .tab");
  var panels={ret:"panel-ret",comp:"panel-comp",rank:"panel-rank"};
  tabs.forEach(function(t){t.addEventListener("click",function(){
    tabs.forEach(function(x){x.classList.remove("active")});t.classList.add("active");
    Object.values(panels).forEach(function(id){document.getElementById(id).classList.remove("active")});
    document.getElementById(panels[t.dataset.panel]).classList.add("active");
  })});
  // Example sub-tabs
  var subtabs=document.querySelectorAll("#exTabs .subtab");
  subtabs.forEach(function(s){s.addEventListener("click",function(){
    subtabs.forEach(function(x){x.classList.remove("active")});s.classList.add("active");
    document.querySelectorAll(".ex-panel").forEach(function(ep){ep.classList.remove("active")});
    document.getElementById("ex-"+s.dataset.ex).classList.add("active");
  })});
  // Accordion
  document.querySelectorAll("#pipeAcc .acc-head").forEach(function(h){
    h.addEventListener("click",function(){h.parentElement.classList.toggle("open")});
  });
  // Copy BibTeX
  var copyBtn=document.getElementById("copyBtn");
  if(copyBtn){copyBtn.addEventListener("click",function(){
    var text=document.getElementById("bibtex").innerText;
    navigator.clipboard.writeText(text).then(function(){
      copyBtn.textContent="Copied";setTimeout(function(){copyBtn.textContent="Copy"},1600);
    });
  })}
  // Back to top
  var toTop=document.getElementById("toTop");
  window.addEventListener("scroll",function(){toTop.classList.toggle("show",window.scrollY>400)},{passive:true});
  toTop.addEventListener("click",function(){window.scrollTo({top:0,behavior:"smooth"})});
  // Pie chart - 12 distinct earthy/muted colors
  var canvas=document.getElementById("pieChart");
  if(canvas){
    var ctx=canvas.getContext("2d");
    var data=[
      {label:"Cell Biology",value:152,color:"#c6773d"},
      {label:"Physics",value:132,color:"#51817f"},
      {label:"Energy Sci.",value:117,color:"#8b6b3e"},
      {label:"Material Sci.",value:116,color:"#b84c3f"},
      {label:"Environ. Sci.",value:116,color:"#4a7c59"},
      {label:"Biology",value:115,color:"#6b5b8a"},
      {label:"Business",value:115,color:"#c49a6c"},
      {label:"Earth Sci.",value:114,color:"#3e5464"},
      {label:"Chemistry",value:113,color:"#9c7b5a"},
      {label:"Math",value:113,color:"#5c6b8a"},
      {label:"Law",value:97,color:"#a0856e"},
      {label:"Astronomy",value:86,color:"#2d5a5a"}
    ];
    var total=data.reduce(function(s,d){return s+d.value},0);
    var dpr=window.devicePixelRatio||1;
    canvas.width=420*dpr;canvas.height=420*dpr;
    canvas.style.width="420px";canvas.style.height="420px";
    ctx.scale(dpr,dpr);
    var cx=206,cy=210,r=130,ir=60;
    var angle=-Math.PI/2;
    data.forEach(function(d){
      var slice=(d.value/total)*Math.PI*2;
      ctx.beginPath();
      ctx.moveTo(cx+Math.cos(angle)*ir,cy+Math.sin(angle)*ir);
      ctx.arc(cx,cy,r,angle,angle+slice);
      ctx.arc(cx,cy,ir,angle+slice,angle,true);
      ctx.closePath();
      ctx.fillStyle=d.color;ctx.fill();
      ctx.strokeStyle="rgba(250,249,247,0.9)";ctx.lineWidth=1.5;ctx.stroke();
      var mid=angle+slice/2;
      var lx=cx+Math.cos(mid)*(r+28);
      var ly=cy+Math.sin(mid)*(r+28);
      ctx.font="600 10px Inter,sans-serif";
      ctx.fillStyle="#4d4b47";
      ctx.textAlign=lx>cx?"left":"right";
      ctx.textBaseline="middle";
      ctx.fillText(d.label,lx,ly-6);
      ctx.font="400 9px Inter,sans-serif";
      ctx.fillStyle="#8c8a85";
      ctx.fillText(d.value,lx,ly+6);
      angle+=slice;
    });
    ctx.font="750 24px Inter,sans-serif";ctx.fillStyle="#1a1a18";
    ctx.textAlign="center";ctx.textBaseline="middle";
    ctx.fillText("1,386",cx,cy-6);
    ctx.font="500 11px Inter,sans-serif";ctx.fillStyle="#8c8a85";
    ctx.fillText("papers",cx,cy+14);
  }
})();
