extern "C" void
cuda_descriptor(unsigned long* mindq,
				float* im1,
				int m,int n,int o,
				int qs,
				int DEV);

void boxfilter(float* input,float* temp1,float* temp2,int hw,int m,int n,int o){
    
    int sz=m*n*o;
    for(int i=0;i<sz;i++){
        temp1[i]=input[i];
    }
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=1;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[(i-1)+j*m+k*m*n];
            }
        }
    }
    
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<(hw+1);i++){
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n];
            }
            for(int i=(hw+1);i<(m-hw);i++){
                temp2[i+j*m+k*m*n]=temp1[(i+hw)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
            }
            for(int i=(m-hw);i<m;i++){
                temp2[i+j*m+k*m*n]=temp1[(m-1)+j*m+k*m*n]-temp1[(i-hw-1)+j*m+k*m*n];
            }
        }
    }
    
    for(int k=0;k<o;k++){
        for(int j=1;j<n;j++){
            for(int i=0;i<m;i++){
                temp2[i+j*m+k*m*n]+=temp2[i+(j-1)*m+k*m*n];
            }
        }
    }
    
    for(int k=0;k<o;k++){
        for(int i=0;i<m;i++){
            for(int j=0;j<(hw+1);j++){
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n];
            }
            for(int j=(hw+1);j<(n-hw);j++){
                temp1[i+j*m+k*m*n]=temp2[i+(j+hw)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
            for(int j=(n-hw);j<n;j++){
                temp1[i+j*m+k*m*n]=temp2[i+(n-1)*m+k*m*n]-temp2[i+(j-hw-1)*m+k*m*n];
            }
        }
    }
    
    for(int k=1;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                temp1[i+j*m+k*m*n]+=temp1[i+j*m+(k-1)*m*n];
            }
        }
    }
    
    for(int j=0;j<n;j++){
        for(int i=0;i<m;i++){
            for(int k=0;k<(hw+1);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n];
            }
            for(int k=(hw+1);k<(o-hw);k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(k+hw)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
            for(int k=(o-hw);k<o;k++){
                input[i+j*m+k*m*n]=temp1[i+j*m+(o-1)*m*n]-temp1[i+j*m+(k-hw-1)*m*n];
            }
        }
    }
    
    
}


void imshift(float* input,float* output,int dx,int dy,int dz,int m,int n,int o){
    for(int k=0;k<o;k++){
        for(int j=0;j<n;j++){
            for(int i=0;i<m;i++){
                if(i+dy>=0&&i+dy<m&&j+dx>=0&&j+dx<n&&k+dz>=0&&k+dz<o)
                output[i+j*m+k*m*n]=input[i+dy+(j+dx)*m+(k+dz)*m*n];
                else
                output[i+j*m+k*m*n]=input[i+j*m+k*m*n];
            }
        }
    }
}

void distances(float* im1,float* d1,int m,int n,int o,int qs,int l){
    int sz1=m*n*o;
	float* w1=new float[sz1];
    int len1=6;

    float* temp1=new float[sz1]; float* temp2=new float[sz1];
    int dx[6]={+qs,+qs,-qs,+0,+qs,+0};
	int dy[6]={+qs,-qs,+0,-qs,+0,+qs};
	int dz[6]={0,+0,+qs,+qs,+qs,+qs};
    
		imshift(im1,w1,dx[l],dy[l],dz[l],m,n,o);
		for(int i=0;i<sz1;i++){
			w1[i]=(w1[i]-im1[i])*(w1[i]-im1[i]);
		}
		boxfilter(w1,temp1,temp2,qs,m,n,o);
		for(int i=0;i<sz1;i++){
			d1[i+l*sz1]=w1[i];
		}
	
    delete[] temp1; delete[] temp2; delete[] w1;
}

//__builtin_popcountll(left[i]^right[i]); absolute hamming distances
void descriptor(unsigned long* mindq,float* im1,int m,int n,int o,int qs){
    image_d=12;
    //MIND with self-similarity context
	    
    //============== DISTANCES USING BOXFILTER ===================
    bool accel = true;
    bool time = false;
    if (accel) {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        cuda_descriptor(mindq,
                        im1,
                        m,n,o,
                        qs,
                        0);
        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_descriptor: "<<timeCOST<<" secs\n";
        }
    } else {
        int dx[6]={+qs,+qs,-qs,+0,+qs,+0};
        int dy[6]={+qs,-qs,+0,-qs,+0,+qs};
        int dz[6]={0,+0,+qs,+qs,+qs,+qs};
        
        int sx[12]={-qs,+0,-qs,+0,+0,+qs,+0,+0,+0,-qs,+0,+0};
        int sy[12]={+0,-qs,+0,+qs,+0,+0,+0,+qs,+0,+0,+0,-qs};
        int sz[12]={+0,+0,+0,+0,-qs,+0,-qs,+0,-qs,+0,-qs,+0};
        int index[12]={0,0,1,1,2,2,3,3,4,4,5,5};
        
        int len1=6;
        const int len2=12;
        int sz1=m*n*o;
        float* d1=new float[sz1*len1];

    #pragma omp parallel for
        for(int l=0;l<len1;l++){
            distances(im1,d1,m,n,o,qs,l);
        }

        //quantisation table
        const int val=6;
        const unsigned long long power=32;
    
#pragma omp parallel for
        for(int k=0;k<o;k++){
            unsigned int tablei[6]={0,1,3,7,15,31};
            float compare[val-1];
            for(int i=0;i<val-1;i++){
                compare[i]=-log((i+1.5f)/val);
            }
            float mind1[12];
            for(int j=0;j<n;j++){
                for(int i=0;i<m;i++){
                    for(int l=0;l<len2;l++){
                        if(i+sy[l]>=0&&i+sy[l]<m&&j+sx[l]>=0&&j+sx[l]<n&&k+sz[l]>=0&&k+sz[l]<o){
                            mind1[l]=d1[i+sy[l]+(j+sx[l])*m+(k+sz[l])*m*n+index[l]*sz1];
                        }
                        else{
                            mind1[l]=d1[i+j*m+k*m*n+index[l]*sz1];
                        }
                    }
                    float minval=*min_element(mind1,mind1+len2);
                    float sumnoise=0.0f;
                    for(int l=0;l<len2;l++){
                        mind1[l]-=minval;
                        sumnoise+=mind1[l];
                    }
                    float noise1=max(sumnoise/(float)len2,1e-6f);
                    for(int l=0;l<len2;l++){
                        mind1[l]/=noise1;
                    }
                    unsigned long long accum=0;
                    unsigned long long tabled1=1;
                    
                    for(int l=0;l<len2;l++)
                    {
                        int mind1val=0;
                        for(int c=0;c<val-1;c++){
                            mind1val+=compare[c]>mind1[l]?1:0;
                        }
                        accum+=tablei[mind1val]*tabled1;
                        tabled1*=power;
                        
                        
                    }
                    mindq[i+j*m+k*m*n]=accum;
                }
            }
        }
    }
}


