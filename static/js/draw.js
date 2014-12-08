function setup() {
    createCanvas(1080,840);
}
function draw(x1,y1,x2,y2) {
    background(255);
    line(x1,y1,x2,y1);
    line(x1,y2,x2,y2);
    line(x2,y1,x1,y1);
    line(x2,y2,x1,y2);
}
