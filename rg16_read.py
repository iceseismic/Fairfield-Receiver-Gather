# Read FairField Receiver Gather 16 format
import numpy as np
import datetime
import matplotlib.pyplot as plt
import struct
from obspy.core import Stream, Trace, UTCDateTime
import os


def readFairFieldRG16(filename, start=None, end=None, head_only=False):
    f = open(filename, 'rb')
    f.seek(10)
    year1 = ord(f.read(1))
    year2 = np.floor(year1/16)
    year3 = year1-year2*16
    year = int(2000+ year2*10+year3)
    
    byte12 = ord(f.read(1))
    genhdrblocks = int(np.floor(byte12/16)+1)
    day_msn = byte12-(genhdrblocks-1)*16
    # print "day_msn:",day_msn
    byte13 = ord(f.read(1))
    # print "byte_13", byte13
    day_lsb1 = np.floor(byte13/16)
    day_lsb2 = byte13-day_lsb1*16
    day_lsb3 = int(day_msn*100+day_lsb1*10+day_lsb2)
    # print day_lsb3
    # print year
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_lsb3-1)

    times = f.read(3)
    times = np.fromstring(times, np.uint8)
    times1 = np.floor(times/16)
    times2 = times-times1*16
    times3 = times1*10 + times2
    # print date
    starttime = date + datetime.timedelta(days=times3[0]/24+times3[1]/(24*60)+times3[2]/(24*60*60))
    # print starttime
    f.seek(6, 1) # 1 is for COF
    sample_rate = ord(f.read(1))/16
    
    f.seek(5, 1)
    channel_sets1 = ord(f.read(1))
    
    channel_sets2 = np.floor(channel_sets1/16)
    channel_sets3 = channel_sets1-channel_sets2*16
    channel_sets = int(channel_sets2*10 + channel_sets3)
    
    f.seek(3, 1)
    
    file_num = struct.unpack('>i', f.read(4))[0] / 256
    
    f.seek(1, 1)
    ext_blocks3 = np.array(struct.unpack('>HH', f.read(4)))
    ext_blocks3[1] = ext_blocks3[1] * 256
    ext_blocks3[1] = ext_blocks3[1] + ord(f.read(1))
    
    f.seek(4, 1)
    record_length = struct.unpack('>i', f.read(4))[0] / 256
    
    f.seek(channel_sets * 32 + 18, 1)
    serial_num = struct.unpack('>i', f.read(4))[0]
    
    #~ f.seek(32*3, 1)
    #~ f.seek(13, 1)
    #~ xcoord = struct.unpack('>i', f.read(4))[0]
    #~ print "xcoord", xcoord
    #~ xcoord= getint(&hdbuf[20+32*4+17],4);
    #~ ycoord= getint(&hdbuf[20+32*4+21],4);
    #~ zcoord= getint(&hdbuf[20+32*4+25],4);
    #~ serial= getint(&hdbuf[20+32],4);
    
    
    f.seek((genhdrblocks+channel_sets+1)*32+16,0)
    num_records = struct.unpack('>i', f.read(4))[0]
    
    # print "t1", f.tell()
    # print "hdr mul:", genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets
    
    f.seek((genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32+30, 0)
    # print "t2", f.tell()
    if channel_sets < 3:
        tmp1 = struct.unpack('>I', f.read(4))[0]*256.
        tmp2 = struct.unpack('>B', f.read(1))[0]
        recline = int(tmp1 + tmp2)
        
        tmp1 = struct.unpack('>I', f.read(4))[0]*256.
        tmp2 = struct.unpack('>B', f.read(1))[0]
        recstation = int(tmp1 + tmp2)
        
    else:
        tmp1 = struct.unpack('>H', f.read(2))[0]*256.
        tmp2 = struct.unpack('>B', f.read(1))[0]
        tmp3 = struct.unpack('>H', f.read(2))[0]/512.
        recline = int(tmp1 + tmp2 + tmp3)
        
        tmp1 = struct.unpack('>H', f.read(2))[0]*256.
        tmp2 = struct.unpack('>B', f.read(1))[0]
        tmp3 = struct.unpack('>H', f.read(2))[0]/512.
        recstation = int(tmp1 + tmp2 + tmp3)
    
    f.seek(12, 1)
    sourceline = struct.unpack('>I', f.read(4))[0]
    sourcestation = struct.unpack('>I', f.read(4))[0]
    
    f.seek(24, 1)
    tr_day = struct.unpack('>Q', f.read(8))[0]/(1e6*60*60*24) # datenum ?
 
    f.seek(254+4*32+16, 0)
    # print struct.unpack('>I', f.read(4))[0]
    
    f.seek((genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32+9, 0)
    ext_trace_blks = ord(f.read(1))
    
    pagain = np.zeros(4)
    for i in range(channel_sets):
        step = (genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32+(20+3*32+8)+(i)*num_records*((20+ext_trace_blks*32)+(record_length/sample_rate*4))
        f.seek(step, 0)
        pagain[i] = struct.unpack('>B', f.read(1))[0]
    
    lcutbinary = np.zeros(4)
    for i in range(channel_sets):
        f.seek((genhdrblocks+i)*32+17,0)
        lcutBCD = struct.unpack('>B', f.read(1))[0]
        lcutBCDms = np.floor(lcutBCD/16)
        lcutBCDls = lcutBCD-lcutBCDms*16
        lcutbinary[i] = lcutBCDms*10+lcutBCDls
    
    f.seek((genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32+9,0)
    ext_trace_blks = ord(f.read(1))
    
    ### READ DATA
    
    num_traces = num_records*channel_sets
    f.seek((genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32,0)
    num_samples= int(record_length/sample_rate)
    data = np.zeros((num_traces, num_samples))
    
    header_length = (genhdrblocks+ext_blocks3[0]+ext_blocks3[1]+channel_sets)*32
    
    traces = []
    for i in range(num_traces):
        f.seek(header_length + i*(20+ext_trace_blks*32) + i*num_samples*4, 0)
        th = f.read(20)
        th01 = f.read(32) #Trace Header 1
        th02 = f.read(32) #Trace Header 2 (shot)
        th03 = f.read(32) #Trace Header 3 (time)
        th04 = f.read(32) #Trace Header 4 (data)
        th05 = f.read(32) #Trace Header 5 (receiver)
        th06 = f.read(32) #Trace Header 6 (orientation)
        th07 = f.read(32) #Trace Header 7 (orientation)
        th08 = f.read(32) #Trace Header 8 (test config)
        th09 = f.read(32) #Trace Header 9 (test config)
        th10 = f.read(32) #Trace Header 10 (test config)
        
        # Processing Time Header (block 3)
        shot, skew_time, corrected_drift, remaining_drift = struct.unpack('>4Q', th03)
        shot = datetime.datetime(1970,1,1) + datetime.timedelta(microseconds=shot)
        if start is None or shot >= start.datetime:
            if end is None or shot <= end.datetime:
                # print "gain:", ord(th04[10]), "dB"
                # print "Trace data starting at", shot
                # Processing Data
                if head_only:
                    rudata = np.zeros(37250)
                else:
                    rudata = np.array(struct.unpack('>%if'%num_samples, f.read(num_samples*4)))
                    
                t = Trace(data=rudata)
                t.stats.sampling_rate = 1.e3 / sample_rate
                t.stats.starttime = shot
                t.stats.station = "%02i.%02i" % (recline, recstation)
                traces.append(t)
        if shot > end.datetime:
            break
    
    s = Stream(traces=traces)
    return s
        
if __name__ == "__main__":
    start = UTCDateTime('2014-07-22Z23:10:00.00')
    end = UTCDateTime('2014-07-22Z23:30:00.00')
    file = r'R11_1.1.0.rg16'
    prim = readFairFieldRG16(file,start=start,end=end)
    print prim
    prim.merge()
    
    file = r'R11_1s.1.0.rg16'
    secon = readFairFieldRG16(file,start=start,end=end)
    print secon
    secon.merge()
    for t in secon:
        t.stats.station += "s"
    
    prim += secon
    prim.plot()
    