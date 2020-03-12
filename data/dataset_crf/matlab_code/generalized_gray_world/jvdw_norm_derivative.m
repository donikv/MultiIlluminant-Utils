function [Rw,Gw,Bw]=jvdw_NormDerivative(in, sigma, order)

if(nargin<3) order=1; end

R=in(:,:,1);
G=in(:,:,2);
B=in(:,:,3);

if(order==1)
    Rx=jvdw_gDer(R,sigma,1,0);
    Ry=jvdw_gDer(R,sigma,0,1);
    Rw=sqrt(Rx.^2+Ry.^2);
    
    Gx=jvdw_gDer(G,sigma,1,0);
    Gy=jvdw_gDer(G,sigma,0,1);
    Gw=sqrt(Gx.^2+Gy.^2);
    
    Bx=jvdw_gDer(B,sigma,1,0);
    By=jvdw_gDer(B,sigma,0,1);
    Bw=sqrt(Bx.^2+By.^2);
end

if(order==2)        %computes frobius norm
    Rxx=jvdw_gDer(R,sigma,2,0);
    Ryy=jvdw_gDer(R,sigma,0,2);
    Rxy=jvdw_gDer(R,sigma,1,1);
    Rw=sqrt(Rxx.^2+4*Rxy.^2+Ryy.^2);
    
    Gxx=jvdw_gDer(G,sigma,2,0);
    Gyy=jvdw_gDer(G,sigma,0,2);
    Gxy=jvdw_gDer(G,sigma,1,1);
    Gw=sqrt(Gxx.^2+4*Gxy.^2+Gyy.^2);
    
    Bxx=jvdw_gDer(B,sigma,2,0);
    Byy=jvdw_gDer(B,sigma,0,2);
    Bxy=jvdw_gDer(B,sigma,1,1);
    Bw=sqrt(Bxx.^2+4*Bxy.^2+Byy.^2);
end
