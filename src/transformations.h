/* several functions to interpolate and symmetrise deformations
 calculates Jacobian and harmonic Energy */

extern "C" void cuda_interp3( float* interp, float* input, float* x1, float* y1, float* z1, int m,int n,int o, int m2,int n2,int o2, bool flag);
extern "C" void cuda_filter1( float *output, float *input, int m, int n, int o, float *filter, int length, int hw, int dim );
extern "C" void cuda_upsample2( float *u1, float *v1, float *w1, float *u0, float *v0, float *w0, int m, int n, int o, int m2, int n2, int o2 );
extern "C" float cuda_jacobian( float *u, float *v, float *w, float *grad, int m, int n, int o, int factor );
extern "C" void cuda_consistentMapping( float *u, float *v, float *w, float *u2, float *v2, float *w2, int m, int n, int o, int factor );


void interp3(float* interp,float* input,float* x1,float* y1,float* z1,int m,int n,int o,int m2,int n2,int o2,bool flag,bool accel)
{
    bool time = false;

    if (accel) {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        cuda_interp3( interp, input,
                      x1, y1, z1,
                      m, n, o,
                      m2, n2, o2,
                      flag );
        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_interp3: "<<timeCOST<<" secs\n";
        }
    }
    else
    {

        for(int k=0;k<o;k++){
            for(int j=0;j<n;j++){
                for(int i=0;i<m;i++)
                {
                    int x=floor(x1[i+j*m+k*m*n]); 
                    int y=floor(y1[i+j*m+k*m*n]);  
                    int z=floor(z1[i+j*m+k*m*n]);
                    float dx=x1[i+j*m+k*m*n]-x; 
                    float dy=y1[i+j*m+k*m*n]-y; 
                    float dz=z1[i+j*m+k*m*n]-z;

                    if(flag)
                    {
                        x+=j; y+=i; z+=k;
                    }
                    interp[i+j*m+k*m*n]=    (1.0-dx)*   (1.0-dy)*   (1.0-dz)*   input[min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                                            (1.0-dx)*   dy*         (1.0-dz)*   input[min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                                            dx*         (1.0-dy)*   (1.0-dz)*   input[min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                                            (1.0-dx)*   (1.0-dy)*   dz*         input[min(max(y,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                                            dx*         dy*         (1.0-dz)*   input[min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z,0),o2-1)*m2*n2]+
                                            (1.0-dx)*   dy*         dz*         input[min(max(y+1,0),m2-1)+min(max(x,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                                            dx*         (1.0-dy)*   dz*         input[min(max(y,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2]+
                                            dx*         dy*         dz*         input[min(max(y+1,0),m2-1)+min(max(x+1,0),n2-1)*m2+min(max(z+1,0),o2-1)*m2*n2];
                }
            }
        }
	}
}



void filter1(float* imagein,float* imageout,int m,int n,int o,float* filter,int length,int dim,bool accel){
	int i,j,k,f;
	int i1,j1,k1;
	int hw=(length-1)/2;
	
	if (accel)
	{
        cuda_filter1( imageout, imagein, m, n, o, filter, length, hw, dim );
    }
    else
    {
        for(i=0;i<(m*n*o);i++)
        {
            imageout[i]=0.0;
        }
        for(k=0;k<o;k++)
        {
            for(j=0;j<n;j++)
            {
                for(i=0;i<m;i++)
                {
                    for(f=0;f<length;f++)
                    {
                        //replicate-padding
                        if(dim==1)
                            imageout[i+j*m+k*m*n]+=filter[f]*imagein[max(min(i+f-hw,m-1),0)+j*m+k*m*n];
                        if(dim==2)
                            imageout[i+j*m+k*m*n]+=filter[f]*imagein[i+max(min(j+f-hw,n-1),0)*m+k*m*n];
                        if(dim==3)
                            imageout[i+j*m+k*m*n]+=filter[f]*imagein[i+j*m+max(min(k+f-hw,o-1),0)*m*n];
                    }
                }
            }
        }
    }
}


// verified
float jacobian(float* u1,float* v1,float* w1,int m,int n,int o,int factor)
{
    float grad[3]={-0.5,0.0,0.5};

    float jstd = 0.f;

	bool accel = true;
    bool time = false;

    if (accel)
    {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        jstd = cuda_jacobian( u1, v1, w1, grad, m, n, o, factor );
        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_jacobian: "<<timeCOST<<" secs\n";
        }
    }
    else
    {
        float factor1=1.0/(float)factor;
        float jmean=0.0;
        int i;

        float* Jac=new float[m*n*o];

        float* J11=new float[m*n*o];
        float* J12=new float[m*n*o];
        float* J13=new float[m*n*o];
        float* J21=new float[m*n*o];
        float* J22=new float[m*n*o];
        float* J23=new float[m*n*o];
        float* J31=new float[m*n*o];
        float* J32=new float[m*n*o];
        float* J33=new float[m*n*o];

        for(i=0;i<(m*n*o);i++)
        {
            J11[i]=0.0;
            J12[i]=0.0;
            J13[i]=0.0;
            J21[i]=0.0;
            J22[i]=0.0;
            J23[i]=0.0;
            J31[i]=0.0;
            J32[i]=0.0;
            J33[i]=0.0;
        }

        float neg=0;
        float Jmin=1;
        float Jmax=1;
        float J;
        float count=0;
        float frac;

        filter1(u1,J11,m,n,o,grad,3,2,accel);
        filter1(u1,J12,m,n,o,grad,3,1,accel);
        filter1(u1,J13,m,n,o,grad,3,3,accel);

        filter1(v1,J21,m,n,o,grad,3,2,accel);
        filter1(v1,J22,m,n,o,grad,3,1,accel);
        filter1(v1,J23,m,n,o,grad,3,3,accel);

        filter1(w1,J31,m,n,o,grad,3,2,accel);
        filter1(w1,J32,m,n,o,grad,3,1,accel);
        filter1(w1,J33,m,n,o,grad,3,3,accel);

        for(i=0;i<(m*n*o);i++)
        {
            J11[i]*=factor1;
            J12[i]*=factor1;
            J13[i]*=factor1;
            J21[i]*=factor1;
            J22[i]*=factor1;
            J23[i]*=factor1;
            J31[i]*=factor1;
            J32[i]*=factor1;
            J33[i]*=factor1;
        }

        for(i=0;i<(m*n*o);i++)
        {
            J11[i]+=1.0;
            J22[i]+=1.0;
            J33[i]+=1.0;
        }
        for(i=0;i<(m*n*o);i++)
        {
            J=      J11[i]*J22[i]*J33[i] +
                    J12[i]*J23[i]*J31[i] +
                    J13[i]*J21[i]*J32[i] -
                    J11[i]*J23[i]*J32[i] -
                    J12[i]*J21[i]*J33[i] -
                    J13[i]*J22[i]*J31[i] ;
            jmean+=J;
            if(J>Jmax)
                Jmax=J;
            if(J<Jmin)
                Jmin=J;
            if(J<0)
                neg++;
            count++;
            Jac[i]=J;
        }
        jmean/=(m*n*o);
        for(int i=0;i<m*n*o;i++)
        {
            jstd+=pow(Jac[i]-jmean,2.0);
        }
        jstd/=(m*n*o-1);
        jstd=sqrt(jstd);
        frac=neg/count;
        cout<<"\nJacobian of deformations| Mean (std): "<<round(jmean*1000)/1000.0<<" ("<<round(jstd*1000)/1000.0<<")\n";
        cout<<"Range: ["<<Jmin<<", "<<Jmax<<"] Negative fraction: "<<frac<<"\n";
        delete []Jac;


        delete []J11;
        delete []J12;
        delete []J13;
        delete []J21;
        delete []J22;
        delete []J23;
        delete []J31;
        delete []J32;
        delete []J33;
    }
	return jstd;
}



void consistentMappingCL(float* u,float* v,float* w,float* u2,float* v2,float* w2,int m,int n,int o,int factor){

	bool accel = true; // true;
    bool time = false;

    if (accel)
    {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        cuda_consistentMapping( u, v, w, u2, v2, w2, m, n, o, factor );
        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_consistentMapping: "<<timeCOST<<" secs\n";
        }
    }
    else
    {
        float factor1=1.0/(float)factor;
        float* us=new float[m*n*o];
        float* vs=new float[m*n*o];
        float* ws=new float[m*n*o];
        float* us2=new float[m*n*o];
        float* vs2=new float[m*n*o];
        float* ws2=new float[m*n*o];

        for(int i=0;i<m*n*o;i++)
        {
            us[i]=u[i]*factor1;
            vs[i]=v[i]*factor1;
            ws[i]=w[i]*factor1;
            us2[i]=u2[i]*factor1;
            vs2[i]=v2[i]*factor1;
            ws2[i]=w2[i]*factor1;
        }

        for(int it=0;it<10;it++)
        {
            interp3(u,us2,us,vs,ws,m,n,o,m,n,o,true,accel);
            interp3(v,vs2,us,vs,ws,m,n,o,m,n,o,true,accel);
            interp3(w,ws2,us,vs,ws,m,n,o,m,n,o,true,accel);

            for(int i=0;i<m*n*o;i++)
            {
                u[i]=0.5*us[i]-0.5*u[i];
                v[i]=0.5*vs[i]-0.5*v[i];
                w[i]=0.5*ws[i]-0.5*w[i];

            }

            interp3(u2,us,us2,vs2,ws2,m,n,o,m,n,o,true,accel);
            interp3(v2,vs,us2,vs2,ws2,m,n,o,m,n,o,true,accel);
            interp3(w2,ws,us2,vs2,ws2,m,n,o,m,n,o,true,accel);

            for(int i=0;i<m*n*o;i++)
            {
                u2[i]=0.5*us2[i]-0.5*u2[i];
                v2[i]=0.5*vs2[i]-0.5*v2[i];
                w2[i]=0.5*ws2[i]-0.5*w2[i];
            }

            for(int i=0;i<m*n*o;i++)
            {
                us[i]=u[i];
                vs[i]=v[i];
                ws[i]=w[i];
                us2[i]=u2[i];
                vs2[i]=v2[i];
                ws2[i]=w2[i];
            }

        }


        for(int i=0;i<m*n*o;i++)
        {
            u[i]*=(float)factor;
            v[i]*=(float)factor;
            w[i]*=(float)factor;
            u2[i]*=(float)factor;
            v2[i]*=(float)factor;
            w2[i]*=(float)factor;
        }

        delete us; delete vs; delete ws;
        delete us2; delete vs2; delete ws2;
    }
}


void upsampleDeformationsCL(float* u1,
                            float* v1,
                            float* w1,
                            float* u0,
                            float* v0,
                            float* w0,
                            int m,int n,int o,
                            int m2,int n2,int o2)
{
    
	bool accel = true; 
    bool time = false;

    if (accel)
    {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        cuda_upsample2( u1, v1, w1,
                        u0, v0, w0,
                        m, n, o,
                        m2, n2, o2 );


        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_upsample2: "<<timeCOST<<" secs\n";
        }
    }
    else
    {
        float scale_m=(float)m/(float)m2;
        float scale_n=(float)n/(float)n2;
        float scale_o=(float)o/(float)o2;

        float* x1=new float[m*n*o];
        float* y1=new float[m*n*o];
        float* z1=new float[m*n*o];
        for(int k=0;k<o;k++)
        {
            for(int j=0;j<n;j++)
            {
                for(int i=0;i<m;i++)
                {
                    x1[i+j*m+k*m*n]=j/scale_n;
                    y1[i+j*m+k*m*n]=i/scale_m;
                    z1[i+j*m+k*m*n]=k/scale_o;
                }
            }
        }

        interp3(u1,u0,x1,y1,z1,m,n,o,m2,n2,o2,false,accel);
        interp3(v1,v0,x1,y1,z1,m,n,o,m2,n2,o2,false,accel);
        interp3(w1,w0,x1,y1,z1,m,n,o,m2,n2,o2,false,accel);

        delete []x1;
        delete []y1;
        delete []z1;

    }
}
