package com.ansj.vec.util;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;

import java.io.*;
import java.util.Collection;
import java.util.TreeSet;

/**
 * Construct Haffman Tree
 * @author ansj
 *
 */
public class Haffman {
    private int layerSize;

    public Haffman(int layerSize) {
        this.layerSize = layerSize;
    }

    private TreeSet<Neuron> set = new TreeSet<>();

    public void make(Collection<Neuron> neurons) {
        set.addAll(neurons);
        while (set.size() > 1) {
            merger();
        }
    }


    private void merger() {
        // TODO Auto-generated method stub
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        hn.freq = min1.freq + min2.freq;
        min1.parent = hn;
        min2.parent = hn;
        min1.code = 0;
        min2.code = 1;
        set.add(hn);
    }

    public Neuron makeWithRoot(Collection<Neuron> neurons) {
        set.addAll(neurons);
        while (set.size() > 1) {
            mergerWithFatherToSon();
        }
        return set.pollFirst();
    }

    private void mergerWithFatherToSon() {
        // TODO Auto-generated method stub
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        hn.freq = min1.freq + min2.freq;
        min1.parent = hn;
        min2.parent = hn;
        min1.code = 0;
        min2.code = 1;
        hn.leftNode = min1;
        hn.rightNode = min2;
        set.add(hn);
    }

    public void make(Collection<Neuron> neurons,File treeFile) throws IOException {
        set.addAll(neurons);
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(treeFile)))){
            while (set.size() > 1) {
                merger(dis);
            }
        }
    }

    private void merger(DataInputStream dis) throws IOException {
        // TODO Auto-generated method stub
        HiddenNeuron hn = new HiddenNeuron(layerSize);
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        hn.freq = min1.freq + min2.freq;
        min1.parent = hn;
        min2.parent = hn;
        min1.code = 0;
        min2.code = 1;
        if(!set.isEmpty()) {
            for (int j = 0; j < layerSize; j++) {
                hn.syn1[j] = dis.readFloat();//System.out.print(hn.syn1[j] + " ");
            }
        }
        set.add(hn);
    }

    public void get(Collection<Neuron> neurons, DataOutputStream dataOutputStream) throws IOException {
        set.addAll(neurons);
        while (set.size() > 1) {
            getParentNode(dataOutputStream);
        }
    }

    private void getParentNode(DataOutputStream dataOutputStream) throws IOException {
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        if(min1 instanceof HiddenNeuron){
            for (double d : ((HiddenNeuron) min1).syn1) {
                dataOutputStream.writeFloat(((Double) d).floatValue());
            }
        }
        if(min2 instanceof HiddenNeuron){
            for (double d : ((HiddenNeuron) min2).syn1) {
                dataOutputStream.writeFloat(((Double) d).floatValue());
            }
        }
        set.add(min1.parent);
    }

    public int getInheritNeurons(Collection<Neuron> neurons) throws IOException {
        set.addAll(neurons);
        int cnt = 0;
        while (set.size() > 1) {
            cnt += getInheritParentNode();
        }
        return cnt;
    }

    private int getInheritParentNode() throws IOException {
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        int cnt = 0;
        if(min1 instanceof HiddenNeuron){
            if(min1.cntAlter && min1.alter)
                cnt++;
        }
        if(min2 instanceof HiddenNeuron){
            if(min2.cntAlter && min2.alter)
                cnt++;
        }
        set.add(min1.parent);
        return cnt;
    }

    public int getInheritNeurons2(Collection<Neuron> neurons) throws IOException {
        set.addAll(neurons);
        int cnt = 0;
        while (set.size() > 1) {
            cnt += getInheritParentNode2();
        }
        return cnt;
    }

    private int getInheritParentNode2() throws IOException {
        Neuron min1 = set.pollFirst();
        Neuron min2 = set.pollFirst();
        int cnt = 0;
        if(min1 instanceof HiddenNeuron){
            if(min1.cnt2Alter && min1.alter)
                cnt++;
        }
        if(min2 instanceof HiddenNeuron){
            if(min2.cnt2Alter && min2.alter)
                cnt++;
        }
        set.add(min1.parent);
        return cnt;
    }
}
