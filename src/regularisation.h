/* Incremental diffusion regularisation of parametrised transformation
 using (globally optimal) belief-propagation on minimum spanning tree.
 Fast distance transform uses squared differences.
 Similarity cost for each node and label has to be given as input.
*/
extern "C" void
cuda_regularisation(    float *u0, float *v0, float *w0,
                        int *ordered, int *parents, float *edgemst,
                        int *startkids, int *numkids, int *kids,
                        float *costall, int *allinds,
                        int *startlev, int *numlev, int maxlev,
                        float quant,
                        int sz1, int len,
                        int DEV );

void messageDT(int ind,float* data,int* indout,int len1,float offsetx,float offsety,float offsetz)
{        
    int len2=len1*len1;
    int len3=len1*len1*len1;
    float z[len1*2+1];
    
    float* val;
    float* valout;
    int* indo;
    
    float* valb;
    float* valb2;
    float buffer[len3];
    float buffer2[len3];
    int* indb;
    int* indb2;
    int bufferi[len3];
    int bufferi2[len3];
    
    for(int i=0;i<len1*2+1;i++){
        z[i]=(i-len1+offsety)*(i-len1+offsety);
    }
    
    for(int k1=0;k1<len1;k1++){
        for(int j1=0;j1<len1;j1++){
            //valb=buffer2+(j1*len1+k1*len1*len1);//
            val=data+ind*len3+(j1*len1+k1*len1*len1);
            valb2=buffer+(j1*len1+k1*len1*len1);
            indb=bufferi+(j1*len1+k1*len1*len1);
            int num=(j1*len1+k1*len1*len1);
            for(int i=0;i<len1;i++){
                float minval=val[0]+z[i+len1];
                int minind=0;
                for(int j=0;j<len1;j++){
                    bool b=(val[j]+z[i-j+len1]<minval);
                    minval=b?val[j]+z[i-j+len1]:minval;
                    minind=b?j:minind;
                }
                valb2[i]=minval;
                indb[i]=minind+num;
            }
            
        }
    }
    for(int i=0;i<len1*2;i++){
        z[i]=(i-len1+offsetx)*(i-len1+offsetx);
    }
    for(int k1=0;k1<len1;k1++){
        for(int i1=0;i1<len1;i1++){
            valb=buffer+(i1+k1*len1*len1);
            valb2=buffer2+(i1+k1*len1*len1);
            indb=bufferi+(i1+k1*len1*len1);
            indb2=bufferi2+(i1+k1*len1*len1);
            for(int i=0;i<len1;i++){
                float minval=valb[0]+z[i+len1];
                int minind=0;
                for(int j=0;j<len1;j++){
                    bool b=(valb[j*len1]+z[i-j+len1]<minval);
                    minval=b?valb[j*len1]+z[i-j+len1]:minval;
                    minind=b?j:minind;
                    
                }
                valb2[i*len1]=minval;
                indb2[i*len1]=indb[minind*len1];
            }
            
        }
    }
    for(int i=0;i<len1*2;i++){
        z[i]=(i-len1+offsetz)*(i-len1+offsetz);
        
    }
    for(int j1=0;j1<len1;j1++){
        for(int i1=0;i1<len1;i1++){
            valb=buffer2+(i1+j1*len1);
            //valb2=buffer+(i1+j1*len1);
            valout=data+ind*len3+(i1+j1*len1);
            indb=bufferi2+(i1+j1*len1);
            //indb2=bufferi+(i1+j1*len1);
            indo=indout+ind*len3+(i1+j1*len1);
            for(int i=0;i<len1;i++){
                float minval=valb[0]+z[i+len1];
                int minind=0;
                for(int j=0;j<len1;j++){
                    bool b=(valb[j*len2]+z[i-j+len1]<minval);
                    minval=b?valb[j*len2]+z[i-j+len1]:minval;
                    minind=b?j:minind;
                }
                valout[i*len2]=minval;
                indo[i*len2]=indb[minind*len2];
            }
            
        }
    }
}

void regularisationCL(  float* costall,
                        float* u0,
                        float* v0,
                        float* w0,
                        float* u1,
                        float* v1,
                        float* w1,
                        int hw,
                        int step1,
                        float quant,
                        int* ordered,
                        int* parents,
                        float* edgemst)
{
    int m2=image_m;
    int n2=image_n;
    int o2=image_o;
            
    int m=m2/step1;
    int n=n2/step1;
    int o=o2/step1;
               
    int sz=m*n*o;
    int len=hw*2+1;

    int len2=len*len*len;
    
    int *selected=new int[sz];
    int *allinds=new int[sz*len2];
    float *cost1=new float[len2];

    int len1=len;
    int len3=len*len*len;

    /////////////////////// NEW //////////////////////////////
    //calculate level boundaries for parallel implementation
    int* levels=new int[sz];
    int* startkids=new int[sz];
    int* numkids=new int[sz];
    int* sortkids=new int[sz];
    int* kids=new int[sz];

    memset(levels, 0, sz * sizeof(int));
    memset(numkids, 0, sz * sizeof(int));
    memset(startkids, 0, sz * sizeof(int));
    memset(sortkids, 0, sz * sizeof(int));
    memset(kids, 0, sz * sizeof(int));
    memset(cost1, 0, len2 * sizeof(float));

    for(int i=0;i<sz;i++){
        int ochild = ordered[i];
        int oparent = parents[ochild];
        numkids[oparent] += 1;
    }
    for(int i=1;i<sz;i++){
        int ochild=ordered[i];
        int oparent=parents[ordered[i]];
        levels[ochild]=levels[oparent]+1;
        startkids[i] = startkids[i-1] + numkids[i-1];
    }
    for(int i=0;i<sz;i++){
        int ochild = ordered[i];
        int oparent = parents[ochild];
        int start = startkids[oparent];
        int sofar = sortkids[oparent];
        kids[start + sofar] = ochild;
        sortkids[oparent] += 1;
    }
    int maxlev=1+*max_element(levels,levels+sz);

    int* numlev=new int[maxlev];
    memset(numlev, 0, maxlev*sizeof(int));
    
    int* startlev=new int[maxlev];
    for(int i=0;i<sz;i++){
        numlev[levels[i]]++;
    }
    startlev[0]=numlev[0];
    for(int i=1;i<maxlev;i++){ //cumulative sum
        startlev[i]=startlev[i-1]+numlev[i];
    }
    delete levels;
    //////////////////////////////////////////////////////////////
    // cout<<"\n\n\nSTARTING REGULARIZATION: \n\n\n"<<flush;
                
    //MAIN LOOP - TO BE PARALLELISED
    //calculate mst-cost
    bool accel = true;
    bool time = false;

    if (accel)
    {
        timeval time1,time2;
        if (time) {
            gettimeofday(&time1, NULL);
        }
        cuda_regularisation(u0, v0, w0,
                            ordered, parents, edgemst,
                            startkids, numkids, kids,
                            costall, allinds,
                            startlev, numlev, maxlev,
                            quant,
                            sz, len,
                            0 );
        if (time) {
            gettimeofday(&time2, NULL);
            float timeCOST=time2.tv_sec+time2.tv_usec/1e6-(time1.tv_sec+time1.tv_usec/1e6);
            cout<<"\n==================================================\n";
            cout<<"Time for cuda_regularisation: "<<timeCOST<<" secs\n";
        }
    } else {
        for(int lev=maxlev-1;lev>0;lev--){
            int start=startlev[lev-1];
            int length=numlev[lev];

            for(int i=start;i<start+length;i++){
                int ochild=ordered[i];
                for(int l=0;l<len2;l++){
                    costall[ochild*len2+l]*=edgemst[ochild];
                }
            }

#pragma omp parallel for
            for(int i=start;i<start+length;i++){
                int ochild=ordered[i];
                int oparent=parents[ordered[i]];
                
                float offsetx=(u0[oparent]-u0[ochild])/(float)quant;
                float offsety=(v0[oparent]-v0[ochild])/(float)quant;
                float offsetz=(w0[oparent]-w0[ochild])/(float)quant;
                messageDT(ochild,costall,allinds,len1,offsetx,offsety,offsetz);
            }

            int start0=startlev[lev-1];
            int length0=numlev[lev];
            for(int i=start0;i<start0+length0;i++){
                int ochild=ordered[i];
                int oparent=parents[ordered[i]];
                float minval=*min_element(costall+ochild*len2,costall+ochild*len2+len3);
                for(int l=0;l<len2;l++){
                    costall[oparent*len2+l]+=(costall[ochild*len2+l]-minval);///edgemst[ochild];//transp
                    //edgemst[ochild]*
                }
            }           
        }
    }

    //mst-cost & select displacement for root note
    int i=0;
    int oroot=ordered[i];
    for(int l=0;l<len2;l++){
        cost1[l]=costall[oroot*len2+l];//transp

    }
    float value=cost1[0]; int index=0;
    
    //dense displacement space
    float* xs=new float[len*len*len];
    float* ys=new float[len*len*len];
    float* zs=new float[len*len*len];
    
    for(int i=0;i<len;i++){
        for(int j=0;j<len;j++){
            for(int k=0;k<len;k++){
                xs[i+j*len+k*len*len]=(j-hw)*quant;
                ys[i+j*len+k*len*len]=(i-hw)*quant;
                zs[i+j*len+k*len*len]=(k-hw)*quant;
            }
        }
    }

    for(int l=0;l<len2;l++){
        if(cost1[l]<value){
            value=cost1[l];
            index=l;
        }
        allinds[oroot*len2+l]=l; //transp

    }
    selected[oroot]=index;
    u1[oroot]=xs[index]+u0[oroot];
    v1[oroot]=ys[index]+v0[oroot];
    w1[oroot]=zs[index]+w0[oroot];


    //select displacements and add to previous deformation field
    for(int i=1;i<sz;i++){
        int ochild=ordered[i];
        int oparent=parents[ordered[i]];
        //select from argmin of based on parent selection
        //index=allinds[ochild+selected[oparent]*sz];
        index=allinds[ochild*len2+selected[oparent]]; //transp
        selected[ochild]=index;
        u1[ochild]=xs[index]+u0[ochild];
        v1[ochild]=ys[index]+v0[ochild];
        w1[ochild]=zs[index]+w0[ochild];
        
    }

    //cout<<"Deformation field calculated!\n";

    delete cost1;
    delete allinds;
    delete selected;
}

