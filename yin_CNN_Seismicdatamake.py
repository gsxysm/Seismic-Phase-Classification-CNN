import pandas as pd
from obspy.core import UTCDateTime
import math 
from obspy import read
from sympy import *
from obspy.signal.trigger import pk_baer
import os
import matplotlib.pyplot as plt
import obspy
from obspy.signal.trigger import recursive_sta_lta, trigger_onset
import numpy 
from obspy.signal.trigger import classic_sta_lta
from obspy.core import read
import scipy.io
#######
f=open('E:/chengxu/pythoncode/huating2.txt',encoding = 'utf-8',errors="ignore")
line=f.readline()#
tmp_list=[]
while line:
    tmp_list.append(line)#
    line=f.readline()
f.close   
out=open('E:/chengxu/pythoncode/处理完1.txt','w')
######
j=1
for i in range(len(tmp_list)):
    if 'eq' in tmp_list[i]:
        print(tmp_list[i])
        event_lat=tmp_list[i][26:32];event_lon=tmp_list[i][34:42];event_dep=tmp_list[i][43:46];
        event_year=tmp_list[i][3:7];event_month=str(int(tmp_list[i][8:10]));event_date=tmp_list[i][11:13];
        event_hour=str(int(tmp_list[i][14:16]));event_min=str(int(tmp_list[i][17:19]));
        event_sec=str(round(float(tmp_list[i][20:25]),2));
        Title='# '+' '+event_year+' '+event_month+' '+event_date+' '+event_hour+' '+\
        event_min+' '+event_sec+' '+event_lat+' '+event_lon+' '+event_dep+' 0'+' 0 '+' 0 '+' 0 '+str(j)+'\n'
        
        eventtime=UTCDateTime(tmp_list[0][3:24]);
        out.write(Title)
        print(Title)
        j=j+1
    else:
        if 'Pg' in tmp_list[i]:
            station_hour=tmp_list[i][32:34];station_min=tmp_list[i][35:37];station_sec=tmp_list[i][38:43];
            if tmp_list[i][3:5]!='  ':#
                      station_name=tmp_list[i][3:8];station_name=station_name.strip()  
            time=UTCDateTime(event_year+'/'+event_month+'/'+event_date+' '+tmp_list[i][32:43]);          
            timeandstation=str(time)+'  '+station_name+'\n'
            out.write(timeandstation)
out.flush()
out.close
'''
time=UTCDateTime(tmp_list[0][3:24]);
st_ = st.select(station="UH1").slice(t, t + 2.5)
event_templates["UH1"] = [st_]
'''
def getRawFileList(path):
     """-------------------------
     files,names=getRawFileList(raw_data_dir)
     files: ['datacn/dialog/one.txt', 'datacn/dialog/two.txt']
     names: ['one.txt', 'two.txt']
     ----------------------------"""
     files = []
     names = []
     for f in os.listdir(path):
         if not f.endswith("~") or not f == "":      
             files.append(os.path.join(path, f))     
             names.append(f)
     return files, names

data_dir='H:/pingliang/2009年09月/'
files,names=getRawFileList(data_dir)
#names=datainput(data_dir);
def sta_lta(trace):
    cft = classic_sta_lta(trace.data, int(2.5 * df), int(10. * df))
    #on_of = trigger_onset(cft, 3.5, 0.5)
    on_of = trigger_onset(cft, 3.5, 0.5)
    
    # Plotting the results
    ax = plt.subplot(211)
    plt.plot(trace.data, 'k')
    ymin, ymax = ax.get_ylim()
    if numpy.size(on_of)>=2:
        plt.vlines(on_of[:, 0], ymin, ymax, color='r', linewidth=2)
        plt.vlines(on_of[:, 1], ymin, ymax, color='b', linewidth=2)
    plt.subplot(212, sharex=ax)
    plt.plot(cft, 'k')
    plt.hlines([3.5, 0.5], 0, len(cft), color=['r', 'b'], linestyle='--')
    plt.axis('tight')
    plt.show()
def fill_0(time):#补0操作
    time=str(time)
    if len(time)==1:
        time='0'+time;
        return time
    else:
        time=time
        return time
f=open('E:/chengxu/pythoncode/处理完1.txt')
line=f.readline()#
tmp_list=[]
while line:
    tmp_list.append(line)
    line=f.readline()
f.close 
#for i in range(len(tmp_list)):
tmp_listout=[]
for i in range(len(tmp_list)):
    if 'T' in tmp_list[i]:
        station_cuttime=UTCDateTime(tmp_list[i][0:27]);
        station_name=tmp_list[i][29:35];
        station_name=station_name.strip()
       # for name in range(len(names)):
        t=str(station_cuttime.year)+fill_0(station_cuttime.month)+fill_0(station_cuttime.day)+str(station_cuttime.hour)+str(station_cuttime.minute)
        T=str(station_cuttime.year)+'-'+fill_0(station_cuttime.month)+'-'+fill_0(station_cuttime.day)
        new_path = data_dir+T+'/'
        if  os.path.exists(new_path) : 
           files,names=getRawFileList(new_path)
           for name in range(len(names)):
            
            if station_name  in names[name] and t in names[name]:
            #if station_name  in names[name]:
            #      if t in names[name]: 
                try:
                        trace=read(new_path+names[name])
                        trace=trace[0]
                       # trace=trace.select(component="Z")
                        trace=trace.slice(station_cuttime-11-8*60*60, station_cuttime + 50-8*60*60)
                        #trace.filter('bandpass', freqmin=0.5, freqmax=2)                       
                        trace.spectrogram(log=True, title='BW.RJOB ' + str(trace.stats.starttime))                       
                        trace.plot()
                       
                        plt.plot(trace.data)
                        plt.save()
                        singlechannel.plot(outfile=name+'.png')
                        
                        #sta_lta(trace)
                        #tmp_listout.append(trace.data)
                        out_data=numpy.hstack((int(t),trace.data))
                        tmp_listout.append(out_data)
                        #scipy.io.savemat('matData.mat',trace.data) 
                except:
                        print(names[name]+' is Wrong data')
                        name=name+1;
                        ######STA/LTA
        else:
            continue
data_out=numpy.array(tmp_listout)
scipy.io.savemat('out.mat', {'data':data_out})
    
   # plt.plot(accelerate, 'k')
#######
# st= tr.select(station="CXT",component="E").slice(t-100, t + 500)
# st.plot()

#for i in range(10,100,2):
#    sta_lta(tr[i])