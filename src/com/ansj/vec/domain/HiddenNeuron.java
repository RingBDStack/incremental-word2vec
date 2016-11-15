package com.ansj.vec.domain;

public class HiddenNeuron extends Neuron{
    
    public double[] syn1 ; //hidden->out
    public Neuron leftNode = null,rightNode = null;//left --> 0  right --> 1

    public HiddenNeuron(int layerSize){
        syn1 = new double[layerSize] ;
    }
}
