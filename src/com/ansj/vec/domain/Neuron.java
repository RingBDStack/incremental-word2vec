package com.ansj.vec.domain;

public abstract class Neuron implements Comparable<Neuron> {
    public int freq;public int freq2;
    public Neuron parent;
    public int code;
    public boolean alter,cnt2Alter,cntAlter,cengAlter;

    @Override
    public int compareTo(Neuron o) {
        // TODO Auto-generated method stub
        if (this.freq > o.freq) {
            return 1;
        } else {
            return -1;
        }
    }

}
