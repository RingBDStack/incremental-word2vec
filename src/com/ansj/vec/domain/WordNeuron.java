package com.ansj.vec.domain;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

public class WordNeuron extends Neuron {
    public String name;
    public double[] syn0 = null; //input->hidden
    public List<Neuron> neurons = null;//neurons on path
    public int[] codeArr = null;
    public List<Neuron> oldNeurons = null;
    public int lowestPublicNode = -1;

    public List<Neuron> makeNeurons(int layerSize) {
        if (neurons != null) {
            return neurons;
        }
        Neuron neuron = this;
        neurons = new LinkedList<>();
        while ((neuron = neuron.parent) != null) {
            neurons.add(neuron);
        }
        Collections.reverse(neurons);
        codeArr = new int[neurons.size()];

        for (int i = 1; i < neurons.size(); i++) {
            codeArr[i - 1] = neurons.get(i).code;
        }
        codeArr[codeArr.length - 1] = this.code;

        return neurons;
    }

    public int makeNeurons2(int layerSize) {
        if (neurons != null) {
            return neurons.size();
        }
        Neuron neuron = this;
        neurons = new LinkedList<>();
        while ((neuron = neuron.parent) != null) {
            neurons.add(neuron);
        }
        Collections.reverse(neurons);
        codeArr = new int[neurons.size()];

        for (int i = 1; i < neurons.size(); i++) {
            codeArr[i - 1] = neurons.get(i).code;
        }
        codeArr[codeArr.length - 1] = this.code;

        return neurons.size();
    }

    public int inheritNeurons(int layerSize , Neuron root) {
        Neuron neuron = this;
        List<Neuron> inputNeurons = new LinkedList<>();
        while ((neuron = neuron.parent) != null) {
            inputNeurons.add(neuron);
        }
        Collections.reverse(inputNeurons);
        int[] inputCodeArr = new int[inputNeurons.size()];

        for (int i = 1; i < inputNeurons.size(); i++) {
            inputCodeArr[i - 1] = inputNeurons.get(i).code;
        }
        inputCodeArr[inputCodeArr.length - 1] = this.code;

        if(neurons == null) {
            //TO DO
            neurons = inputNeurons;
            codeArr = inputCodeArr;
            return 0;
        }
        int cnt = 0;

        oldNeurons = new LinkedList<>();
        HiddenNeuron hn = null,oldHn = null;
        boolean flag = true;
        int len = 0;

        for(int i = 0;i < codeArr.length;++i){
            if(root instanceof WordNeuron) {
                break;
            }
            if(flag && i < inputCodeArr.length && inputCodeArr[i] == codeArr[i])
                lowestPublicNode = i;
            else {
                flag = false;
                this.alter = true;
            }
            if(!root.alter){
                oldHn = (HiddenNeuron) neurons.get(i);
                hn = (HiddenNeuron)root;
                for (int j = 0; j < layerSize; ++j) {
                    if (!Double.isNaN(oldHn.syn1[j]))
                        hn.syn1[j] = oldHn.syn1[j];
                }
                oldHn.syn1 = null;
                cnt++;
                root.alter = true;
            }

            oldNeurons.add(root);

            if(codeArr[i] == 0)
                root = ((HiddenNeuron)root).leftNode;
            else
                root = ((HiddenNeuron)root).rightNode;
        }
        for(int i = 0;i < inputNeurons.size();++i){
            Neuron x = inputNeurons.get(i);
            if(!x.cengAlter){
                x.cengAlter = true;
            }
        }
        neurons = inputNeurons;
        codeArr = inputCodeArr;
        return cnt;
    }

    public int inheritNeuronsCount(int layerSize , Neuron root) {
        Neuron neuron = this;
        List<Neuron> inputNeurons = new LinkedList<>();
        while ((neuron = neuron.parent) != null) {
            inputNeurons.add(neuron);
        }
        Collections.reverse(inputNeurons);
        int[] inputCodeArr = new int[inputNeurons.size()];

        for (int i = 1; i < inputNeurons.size(); i++) {
            inputCodeArr[i - 1] = inputNeurons.get(i).code;
        }
        inputCodeArr[inputCodeArr.length - 1] = this.code;

        if(neurons == null) {
            //TO DO
            neurons = inputNeurons;
            codeArr = inputCodeArr;
            return 0;
        }
        int cnt = 0;

        oldNeurons = new LinkedList<>();
        HiddenNeuron hn = null,oldHn = null;
        boolean flag = true;
        int longer = codeArr.length - inputCodeArr.length;
        for(int i = 0;i < codeArr.length;++i){
            if(root instanceof WordNeuron) {
                break;
            }
            if(flag && i < inputCodeArr.length && inputCodeArr[i] == codeArr[i])
                lowestPublicNode = i;
            else {
                flag = false;
                this.alter = true;
            }
            if(i >= inputCodeArr.length){
                root.cntAlter = true;
            }
            if(!root.alter){
                oldHn = (HiddenNeuron) neurons.get(i);
                hn = (HiddenNeuron)root;
                for (int j = 0; j < layerSize; ++j) {
                    if (!Double.isNaN(oldHn.syn1[j]))
                        hn.syn1[j] = oldHn.syn1[j];
                }
                oldHn.syn1 = null;
//                cnt++;
                root.alter = true;
            }
            oldNeurons.add(root);

            if(codeArr[i] == 0)
                root = ((HiddenNeuron)root).leftNode;
            else
                root = ((HiddenNeuron)root).rightNode;
        }
        if(longer < 0){
            for(int i = inputCodeArr.length + longer ; i < inputCodeArr.length ;i++){
                if(inputNeurons.get(i) instanceof WordNeuron) {
                    break;
                }
                hn = (HiddenNeuron)inputNeurons.get(i);
                hn.cnt2Alter = true;
            }
        }
        neurons = inputNeurons;
        codeArr = inputCodeArr;
        return cnt;
    }
    public WordNeuron(String name, int freq, int layerSize) {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        Random random = new Random();
        for (int i = 0; i < syn0.length; i++) {
            syn0[i] = (random.nextDouble() - 0.5) / layerSize;
        }
    }

    public WordNeuron(String name, int freq, int layerSize, DataInputStream dis) throws IOException {
        this.name = name;
        this.freq = freq;
        this.syn0 = new double[layerSize];
        for (int i = 0; i < syn0.length; i++) {
            syn0[i] = dis.readFloat();
        }
    }
}