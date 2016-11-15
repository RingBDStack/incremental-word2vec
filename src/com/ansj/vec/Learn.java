package com.ansj.vec;

import com.ansj.vec.domain.HiddenNeuron;
import com.ansj.vec.domain.Neuron;
import com.ansj.vec.domain.WordNeuron;
import com.ansj.vec.util.Haffman;
import com.ansj.vec.util.MapCount;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Learn {

    private Map<String, Neuron> wordMap = new HashMap<>();
    /**
     * 训练多少个特征
     */
    private int layerSize = 50;

    /**
     * 上下文窗口大小
     */
    private int window = 5;

    private double sample = 1e-4;
    private double alpha = 0.025;
    private double startingAlpha = alpha;

    private int EXP_TABLE_SIZE = 1000;

    private Boolean isCbow = false;

    private double[] expTable = new double[EXP_TABLE_SIZE];

    private int trainWordsCount = 0;

    private int MAX_EXP = 6;

    private LinkedList<Tri> taskList = new LinkedList<>();

    private int MAX_SIZE = 5000;

    private int threadSize = 20;

    public Learn(Boolean isCbow, Integer layerSize, Integer window, Double alpha, Double sample) {
        createExpTable();
        if (isCbow != null) {
            this.isCbow = isCbow;
        }
        if (layerSize != null)
            this.layerSize = layerSize;
        if (window != null)
            this.window = window;
        if (alpha != null)
            this.alpha = alpha;
        if (sample != null)
            this.sample = sample;
    }

    public Learn() {
        createExpTable();
    }

    /**
     * trainModel Globally
     * @throws java.io.IOException
     */
    private void trainModel(File file) throws IOException {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(threadSize, threadSize, threadSize, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<Runnable>(threadSize));
        for(int i = 0;i < threadSize;++i) {
            MyTask myTask = new MyTask();
            executor.execute(myTask);
        }

        try (BufferedReader br = new BufferedReader(
            new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            long nextRandom = 5;
            int wordCount = 0;
            int lastWordCount = 0;
            int wordCountActual = 0;
            synchronized (taskList) {
                while ((temp = br.readLine()) != null) {
                    if (wordCount - lastWordCount > 10000) {
//                    System.out.println("alpha:" + alpha + "\tProgress: "
//                                 + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
//                                 + "%");
                        wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                        if (alpha < startingAlpha * 0.0001) {
                            alpha = startingAlpha * 0.0001;
                        }
                    }
                    String[] strs = temp.split(" ");
                    wordCount += strs.length;
                    List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                    for (int i = 0; i < strs.length; i++) {
                        Neuron entry = wordMap.get(strs[i]);
                        if (entry == null) {
                            continue;
                        }
                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0) {
                            double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                    * (sample * trainWordsCount) / entry.freq;
                            nextRandom = nextRandom * 25214903917L + 11;
                            if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                                continue;
                            }
                        }
                        sentence.add((WordNeuron) entry);
                    }

                    while (taskList.size() > MAX_SIZE)
                    {
                        try
                        {
                            taskList.wait();
                        }
                        catch (InterruptedException e)
                        {
                            e.printStackTrace();
                        }
                    }
                    taskList.add(new Tri(nextRandom,sentence,(short)2));
                    taskList.notifyAll();
//                    for (int index = 0; index < sentence.size(); index++) {
//                        nextRandom = nextRandom * 25214903917L + 11;
//                        if (isCbow) {
//                            cbowGram(index, sentence, (int) nextRandom % window);
//                        } else {
//                            skipGram(index, sentence, (int) nextRandom % window);
//                        }
//                    }
                }
//            System.out.println("Vocab size: " + wordMap.size());
//            System.out.println("Words in train file: " + trainWordsCount);
//            System.out.println("sucess train over!");
            }
        }
        executor.shutdown();
    }

    private void trainModelBlindly(File file, File fileAdded) throws IOException {
        String temp = null;
        long nextRandom = 5;
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(file)))) {
            while ((temp = br.readLine()) != null) {
                if (wordCount - lastWordCount > 10000) {
                    wordCountActual += wordCount - lastWordCount;
                    lastWordCount = wordCount;
                    alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                    if (alpha < startingAlpha * 0.0001) {
                        alpha = startingAlpha * 0.0001;
                    }
                }
                String[] strs = temp.split(" ");
                wordCount += strs.length;
                List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                for (int i = 0; i < strs.length; i++) {
                    Neuron entry = wordMap.get(strs[i]);
                    if (entry == null) {
                        continue;
                    }
                    // The subsampling randomly discards frequent words while keeping the ranking same
                    if (sample > 0) {
                        double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                * (sample * trainWordsCount) / entry.freq;
                        nextRandom = nextRandom * 25214903917L + 11;
                        if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                            continue;
                        }
                    }
                    sentence.add((WordNeuron) entry);
                }

                for (int index = 0; index < sentence.size(); index++) {
                    nextRandom = nextRandom * 25214903917L + 11;
                    if (isCbow) {
                        cbowGram(index, sentence, (int) nextRandom % window);
                    } else {
                        skipGram(index, sentence, (int) nextRandom % window);
                    }
                }

            }
        }
        long start = System.currentTimeMillis();
        int xx = 0;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileAdded)))) {
            while ((temp = br.readLine()) != null) {
                    if (wordCount - lastWordCount > 10000) {
//                        System.out.println("alpha:" + alpha + "\tProgress: "
//                                        + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
//                                        + "%");
                        wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                        if (alpha < startingAlpha * 0.0001) {
                            alpha = startingAlpha * 0.0001;
                        }
                    }
                    String[] strs = temp.split(" ");
                    wordCount += strs.length;
                    List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                    for (int i = 0; i < strs.length; i++) {
                        Neuron entry = wordMap.get(strs[i]);
                        if (entry == null) {
                            continue;
                        }
                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0) {
                            double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                    * (sample * trainWordsCount) / entry.freq;
                            nextRandom = nextRandom * 25214903917L + 11;
                            if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                                continue;
                            }
                        }
                        sentence.add((WordNeuron) entry);
                    }

                    for (int index = 0; index < sentence.size(); index++) {
                        nextRandom = nextRandom * 25214903917L + 11;
                        if (isCbow) {
                            cbowGram(index, sentence, (int) nextRandom % window);
                        } else {
                            skipGram(index, sentence, (int) nextRandom % window);
                        }
                    }
                xx++;
            }
        }
        System.out.println((System.currentTimeMillis() - start));
    }

    private void trainModelThirdType(File file, File fileAdded) throws IOException {
        String temp = null;
        long nextRandom = 5;
        int wordCount = 0;
        int lastWordCount = 0;
        int wordCountActual = 0;

        ThreadPoolExecutor executor = new ThreadPoolExecutor(threadSize, threadSize, threadSize, TimeUnit.MILLISECONDS,
                new ArrayBlockingQueue<Runnable>(threadSize));
        for(int i = 0;i < threadSize;++i) {
            MyTaskThirdType myTask = new MyTaskThirdType();
            executor.execute(myTask);
        }

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(file)))) {
            synchronized (taskList) {
                while ((temp = br.readLine()) != null) {
                    if (wordCount - lastWordCount > 10000) {
//                        System.out.println("alpha:" + alpha + "\tProgress: "
//                                + (int) (wordCountActual / (double) (trainWordsCount + 1) * 100)
//                                + "%");
                        wordCountActual += wordCount - lastWordCount;
                        lastWordCount = wordCount;
                        alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                        if (alpha < startingAlpha * 0.0001) {
                            alpha = startingAlpha * 0.0001;
                        }
                    }

                    String[] strs = temp.split(" ");
//                    wordCount += strs.length;
                    List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                    for (int i = 0; i < strs.length; i++) {
                        Neuron entry = wordMap.get(strs[i]);
                        if (entry == null) {
                            continue;
                        }
                        // The subsampling randomly discards frequent words while keeping the ranking same
                        if (sample > 0) {
                            double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                    * (sample * trainWordsCount) / entry.freq;
                            nextRandom = nextRandom * 25214903917L + 11;
                            if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                                continue;
                            }
                        }
                        sentence.add((WordNeuron) entry);
                    }
                    wordCount += sentence.size();
                    if(sentence.isEmpty())continue;
                    while (taskList.size() > MAX_SIZE)
                    {
                        try
                        {
                            taskList.wait();
                        }
                        catch (InterruptedException e)
                        {
                            e.printStackTrace();
                        }
                    }
                    taskList.add(new Tri(nextRandom,sentence,(short)0));
                    taskList.notifyAll();
                }
            }
        }

        int xx = 0;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(fileAdded)))) {
            synchronized (taskList) {
                while ((temp = br.readLine()) != null) {
                        if (wordCount - lastWordCount > 10000) {
                            wordCountActual += wordCount - lastWordCount;
                            lastWordCount = wordCount;
                            alpha = startingAlpha * (1 - wordCountActual / (double) (trainWordsCount + 1));
                            if (alpha < startingAlpha * 0.0001) {
                                alpha = startingAlpha * 0.0001;
                            }
                        }
                        String[] strs = temp.split(" ");
                        wordCount += strs.length;
                        List<WordNeuron> sentence = new ArrayList<WordNeuron>();
                        for (int i = 0; i < strs.length; i++) {
                            Neuron entry = wordMap.get(strs[i]);
                            if (entry == null) {
                                continue;
                            }
                            // The subsampling randomly discards frequent words while keeping the ranking same
                            if (sample > 0) {
                                double ran = (Math.sqrt(entry.freq / (sample * trainWordsCount)) + 1)
                                        * (sample * trainWordsCount) / entry.freq;
                                nextRandom = nextRandom * 25214903917L + 11;
                                if (ran < (nextRandom & 0xFFFF) / (double) 65536) {
                                    continue;
                                }
                            }
                            sentence.add((WordNeuron) entry);
                        }

                        while (taskList.size() > MAX_SIZE) {
                            try {
                                taskList.wait();
                            } catch (InterruptedException e) {
                                e.printStackTrace();
                            }
                        }
                        taskList.add(new Tri(nextRandom, sentence, (short) 2));
                        taskList.notifyAll();
                    xx++;
                }
            }
        }
        executor.shutdown();
    }

    /**
     * skip gram 模型训练
     * @param sentence
     */
    private void skipGram(int index, List<WordNeuron> sentence, int b) {
        // TODO Auto-generated method stub
        WordNeuron word = sentence.get(index);
        // TODO Auto-generated method stub
        int a, c = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) {
                continue;
            }

            double[] neu1e = new double[layerSize];//误差项
            //HIERARCHICAL SOFTMAX
            List<Neuron> neurons = word.neurons;
            WordNeuron we = sentence.get(c);
            for (int i = 0; i < neurons.size(); i++) {
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
                if(out.syn1 == null) {
//                    System.out.print(we.name + " out ");
                    out.syn1 = new double[layerSize];
                }
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.syn0[j] * out.syn1[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // 'g' is the gradient multiplied by the learning rate
                double g = (1 - word.codeArr[i] - f) * alpha;
                // Propagate errors output -> hidden
                for (c = 0; c < layerSize; c++) {
                    neu1e[c] += g * out.syn1[c];
//                    we.splitSyn0.get(i)[c] = g * out.syn1[c];
                }
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
            }

            // Learn weights input -> hidden
            for (int j = 0; j < layerSize; j++) {
                we.syn0[j] += neu1e[j];
            }
        }

    }

    /**
     * 词袋模型
     * @param index
     * @param sentence
     * @param b
     */
    private void cbowGram(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        int a, c = 0;

        List<Neuron> neurons = word.neurons;
        double[] neu1e = new double[layerSize];//误差项
        double[] neu1 = new double[layerSize];//误差项
        WordNeuron last_word;

        for (a = b; a < window * 2 + 1 - b; a++)
            if (a != window) {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    neu1[c] += last_word.syn0[c];
            }

        //HIERARCHICAL SOFTMAX
        for (int d = 0; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
            if(out.syn1 == null) {
//                    System.out.print(we.name + " out ");
                out.syn1 = new double[layerSize];
            }
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++)
                f += neu1[c] * out.syn1[c];
            if (f <= -MAX_EXP)
                continue;
            else if (f >= MAX_EXP)
                continue;
            else
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            //            double g = (1 - word.codeArr[d] - f) * alpha;
            //              double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;
            //
            for (c = 0; c < layerSize; c++) {
                neu1e[c] += g * out.syn1[c];
                //we.splitSyn0.get(i)[c] = g * out.syn1[c];
            }
            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
        }
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a != window) {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    last_word.syn0[c] += neu1e[c];
            }

        }
    }

    private void skipGram_Incrementally(int index, List<WordNeuron> sentence, int b) {
        // TODO Auto-generated method stubS
        WordNeuron word = sentence.get(index);
        if(!word.alter)return;
        int a, c = 0;
        for (a = b; a < window * 2 + 1 - b; a++) {
            if (a == window) {
                continue;
            }
            c = index - window + a;
            if (c < 0 || c >= sentence.size()) {
                continue;
            }

            //HIERARCHICAL SOFTMAX
            List<Neuron> neurons = word.neurons;
            List<Neuron> oldNeurons = word.oldNeurons;
            WordNeuron we = sentence.get(c);

            for (int i = word.lowestPublicNode + 1; ; i++) {
                if(i >= neurons.size())
                    break;
                HiddenNeuron out = (HiddenNeuron) neurons.get(i);
//                if(!out.alter)continue;
                double f = 0;
                // Propagate hidden -> output
                for (int j = 0; j < layerSize; j++) {
                    f += we.syn0[j] * out.syn1[j];
                }
                if (f <= -MAX_EXP || f >= MAX_EXP) {
                    continue;
                } else {
                    f = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
                    f = expTable[(int) f];
                }
                // 'g' is the gradient multiplied by the learning rate
                double g = (1 - word.codeArr[i] - f) * alpha;
                // Learn weights hidden -> output
                for (c = 0; c < layerSize; c++) {
                    out.syn1[c] += g * we.syn0[c];
                }
                // Undo Learn weights hidden -> output
                if(i < oldNeurons.size()){
                    HiddenNeuron oldOut = (HiddenNeuron) oldNeurons.get(i);
                    for (c = 0; c < layerSize; c++) {
                        oldOut.syn1[c] -= g * we.syn0[c];
                    }
                }
            }
        }

    }

    private void cbowGram_Incrementally(int index, List<WordNeuron> sentence, int b) {
        WordNeuron word = sentence.get(index);
        if(!word.alter)
            return;
        int a, c = 0;

        List<Neuron> neurons = word.neurons;
        List<Neuron> oldNeurons = word.oldNeurons;
        double[] neu1 = new double[layerSize];//误差项
        WordNeuron last_word;

        for (a = b; a < window * 2 + 1 - b; a++)
            if (a != window) {
                c = index - window + a;
                if (c < 0)
                    continue;
                if (c >= sentence.size())
                    continue;
                last_word = sentence.get(c);
                if (last_word == null)
                    continue;
                for (c = 0; c < layerSize; c++)
                    neu1[c] += last_word.syn0[c];
            }

        //HIERARCHICAL SOFTMAX
        for (int d = word.lowestPublicNode+1; d < neurons.size(); d++) {
            HiddenNeuron out = (HiddenNeuron) neurons.get(d);
//            if (!out.alter) continue;
            double f = 0;
            // Propagate hidden -> output
            for (c = 0; c < layerSize; c++)
                f += neu1[c] * out.syn1[c];
            if (f <= -MAX_EXP)
                continue;
            else if (f >= MAX_EXP)
                continue;
            else
                f = expTable[(int) ((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            // 'g' is the gradient multiplied by the learning rate
            //            double g = (1 - word.codeArr[d] - f) * alpha;
            //              double g = f*(1-f)*( word.codeArr[i] - f) * alpha;
            double g = f * (1 - f) * (word.codeArr[d] - f) * alpha;

            // Learn weights hidden -> output
            for (c = 0; c < layerSize; c++) {
                out.syn1[c] += g * neu1[c];
            }
            if(d < oldNeurons.size()){
                HiddenNeuron oldOut = (HiddenNeuron) oldNeurons.get(d);
                for (c = 0; c < layerSize; c++) {
                    oldOut.syn1[c] -= g * neu1[c];
                }
            }
        }
    }

    /**
     * 统计词频
     * @param file
     * @throws java.io.IOException
     */
    private void readVocab(File file) throws IOException {
        MapCount<String> mc = new MapCount<>();
        try (BufferedReader br = new BufferedReader(
            new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            while ((temp = br.readLine()) != null) {
                String[] split = temp.split(" ");
                trainWordsCount += split.length;
                for (String string : split) {
                    mc.add(string);
                }
            }
        }
        for (Entry<String, Integer> element : mc.get().entrySet()) {
            wordMap.put(element.getKey(), new WordNeuron(element.getKey(), element.getValue(),
                layerSize));
        }
    }

    /**
     * 添加词频
     * @param file
     * @throws java.io.IOException
     */
    private void addVocab(File file) throws IOException {
        MapCount<String> mc = new MapCount<>();int xx = 0;int cntBytes = 0;
        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(file)))) {
            String temp = null;
            while ((temp = br.readLine()) != null) {
                    cntBytes += temp.getBytes().length;

                    String[] split = temp.split(" ");
                    trainWordsCount += split.length;
                    for (String string : split) {
                        mc.add(string);
                    }
                xx++;
            }
        }
        for (Entry<String, Integer> element : mc.get().entrySet()) {
            if(wordMap.containsKey(element.getKey())){
//                System.out.print(element.getKey() + ":" + wordMap.get(element.getKey()).freq + "+" + element.getValue());
                wordMap.get(element.getKey()).freq += element.getValue();
//                System.out.println("="+wordMap.get(element.getKey()).freq);
            }
            else {
//                System.out.println(element.getKey()+":"+element.getValue()+"(new word)");
                wordMap.put(element.getKey(), new WordNeuron(element.getKey(), element.getValue(),
                        layerSize));
            }
        }
    }

    /**
     * 统计词频 & 预置词向量
     * @param modelFile
     * @throws java.io.IOException
     */
    private void readVocabFromModelPlus(File modelFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(modelFile)))) {
            int size = dis.readInt();
            this.layerSize = dis.readInt();
            this.trainWordsCount = dis.readInt();
            String key = null;
            int val = 0;
            for (int i = 0;i < size;++i) {
                key = dis.readUTF();
                val = dis.readInt();
                wordMap.put(key, new WordNeuron(key, val,layerSize,dis));
            }
        }

    }

    /**
     * 统计词频 & 重置词向量
     * @param modelFile
     * @throws java.io.IOException
     */
    private void readVocabFromModelAndReset(File modelFile) throws IOException {
        try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(modelFile)))) {
            int size = dis.readInt();
            this.layerSize = dis.readInt();
            this.trainWordsCount = dis.readInt();
            String key = null;
            int val = 0;
            for (int i = 0;i < size;++i) {
                key = dis.readUTF();
                val = dis.readInt();
                wordMap.put(key, new WordNeuron(key, val,layerSize));
                for(int j = 0;j < layerSize;++j)
                    dis.readFloat();
            }
        }
    }
    /**
     * Precompute the exp() table
     * f(x) = x / (x + 1)
     */
    private void createExpTable() {
        for (int i = 0; i < EXP_TABLE_SIZE; i++) {
            // exp(6 * ((i-500)/500)) => e^-6~e^6
            expTable[i] = Math.exp(((i / (double) EXP_TABLE_SIZE * 2 - 1) * MAX_EXP));
            // e^6/(e^6 + 1) = 1/(1+e^-6)
            expTable[i] = expTable[i] / (expTable[i] + 1);
        }
    }

    /**
     * 根据文件学习
     * @param file
     * @throws java.io.IOException
     */
    public void learnFile(File file) throws IOException {
        readVocab(file);
        long start = System.currentTimeMillis();
        new Haffman(layerSize).make(wordMap.values());

        //查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron)neuron).makeNeurons(layerSize) ;
        }
        System.out.println("wordMap.size= "+wordMap.size());
        trainModel(file);
    }

    /**
     * 根据file建树,用file&fileAdded学习
     * @param file
     * @throws java.io.IOException
     */
    public void learnFileBlindly(File file,File fileAdded) throws IOException {
        readVocab(file);
        long start = System.currentTimeMillis();
        new Haffman(layerSize).make(wordMap.values());

        //查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron)neuron).makeNeurons(layerSize) ;
        }
        System.out.print((System.currentTimeMillis() - start) + " ");
        trainModelBlindly(file, fileAdded);
    }

    /**
     * 根据文件增量学习
     * @param file,fileAdded,treeFile,leaveFile
     * @throws java.io.IOException
     */
    public void learnFile_Incrementally(File file, File fileAdded, File treeFile, File modelFile) throws IOException {
        //还原词向量 & 二叉树
        readVocabFromModelPlus(modelFile);
        new Haffman(layerSize).make(wordMap.values(),treeFile);
        //查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron)neuron).makeNeurons2(layerSize);
        }

        long cnt = 0;
        //增量构造二叉树
        addVocab(fileAdded);
        Neuron root = new Haffman(layerSize).makeWithRoot(wordMap.values());
        //比对每个神经元
        for (Neuron neuron : wordMap.values()) {
            cnt += ((WordNeuron)neuron).inheritNeurons(layerSize,root) ;
        }
        System.out.println(":"+cnt + "/" + wordMap.size()+"="+((double) cnt / wordMap.size()));
        long start = System.currentTimeMillis();
        trainModelThirdType(file, fileAdded);
        System.out.println((System.currentTimeMillis() - start));
    }

    public void learnFile_Incrementally_Count(File file,File fileAdded,File treeFile,File modelFile) throws IOException {
        //还原词向量 & 二叉树
        readVocabFromModelPlus(modelFile);
        new Haffman(layerSize).make(wordMap.values(),treeFile);
        //查找每个神经元
        for (Neuron neuron : wordMap.values()) {
            ((WordNeuron)neuron).makeNeurons(layerSize) ;
        }

        int cnt = 0,cnt2 = 0;
        //增量构造二叉树
        addVocab(fileAdded);
        Neuron root = new Haffman(layerSize).makeWithRoot(wordMap.values());
        //比对每个神经元
        for (Neuron neuron : wordMap.values()) {
            cnt2 += ((WordNeuron)neuron).inheritNeuronsCount(layerSize,root) ;
//            cnt += ((WordNeuron)neuron).inheritNeurons(layerSize,root) ;
        }
        cnt = new Haffman(layerSize).getInheritNeurons(wordMap.values());
        cnt2 = new Haffman(layerSize).getInheritNeurons2(wordMap.values());
//        int cnt2 = 0;
//        for (Neuron neuron : wordMap.values()) {
//            if(neuron.alter)
//                cnt2++;
//        }
        System.out.println("变长节点: "+cnt + "/" + wordMap.size() + "=" + ((double) cnt / wordMap.size()));
        System.out.println("变短节点: "+cnt2 + "/" + wordMap.size() + "=" + ((double) cnt2 / wordMap.size()));
//        __out.write("非叶子节点: " + cnt + "/" + wordMap.size() + "=" + ((double) cnt / wordMap.size()) + "\r\n");
//        __out.write("叶子节点: "+cnt2 + "/" + wordMap.size() + "=" + ((double) cnt2 / wordMap.size()) +"\r\n");
//        __out.write(((double) cnt / wordMap.size()) + " " + ((double) cnt2 / wordMap.size()) + " ");
//        System.out.print(((double) cnt / wordMap.size()) + " " + ((double) cnt2 / wordMap.size()) + " ");
//        System.exit(-1);
    }

    /**
     * 保存模型
     */
    public void saveModel(File file) {
        // TODO Auto-generated method stub

        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(file)))) {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn0 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    /**
     * 保存模型+词频
     */
    public void saveModelPlus(File file) {
        // TODO Auto-generated method stub

        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(file)))) {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            dataOutputStream.writeInt(trainWordsCount);
            double[] syn0 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                dataOutputStream.writeInt(element.getValue().freq);
                syn0 = ((WordNeuron) element.getValue()).syn0;
                for (double d : syn0) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    /*
    保存二叉树非叶子节点
     */
    public void saveTreeNodes(File file) {
        // TODO Auto-generated method stub

        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(file)))) {
//            dataOutputStream.writeInt(wordMap.size());
//            dataOutputStream.writeInt(layerSize);
            new Haffman(layerSize).get(wordMap.values(), dataOutputStream);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }
    /*
    保存叶子节点,useless
     */
    public void saveLeaveNodes(File file){
        // TODO Auto-generated method stub

        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(file)))) {
//            dataOutputStream.writeInt(wordMap.size());
//            dataOutputStream.writeInt(layerSize);
            int[] codeArr = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                codeArr = ((WordNeuron) element.getValue()).codeArr;
                dataOutputStream.writeInt(codeArr.length);
                for (int d : codeArr) {
                    dataOutputStream.writeInt(d);
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    /*
    保存最底层非叶子节点,useless
     */
    public void saveTheta(File file) {
        // TODO Auto-generated method stub

        try (DataOutputStream dataOutputStream = new DataOutputStream(new BufferedOutputStream(
                new FileOutputStream(file)))) {
            dataOutputStream.writeInt(wordMap.size());
            dataOutputStream.writeInt(layerSize);
            double[] syn1 = null;
            for (Entry<String, Neuron> element : wordMap.entrySet()) {
                dataOutputStream.writeUTF(element.getKey());
                WordNeuron wordNeuron = (WordNeuron)element.getValue();
                HiddenNeuron hiddenNeuron = (HiddenNeuron) wordNeuron.neurons.get(wordNeuron.neurons.size() - 1);
                syn1 = hiddenNeuron.syn1;
                if(syn1 == null)syn1 = new double[layerSize];
                for (double d : syn1) {
                    dataOutputStream.writeFloat(((Double) d).floatValue());
                }
            }
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public int getLayerSize() {
        return layerSize;
    }

    public void setLayerSize(int layerSize) {
        this.layerSize = layerSize;
    }

    public int getWindow() {
        return window;
    }

    public void setWindow(int window) {
        this.window = window;
    }

    public double getSample() {
        return sample;
    }

    public void setSample(double sample) {
        this.sample = sample;
    }

    public double getAlpha() {
        return alpha;
    }

    public void setAlpha(double alpha) {
        this.alpha = alpha;
        this.startingAlpha = alpha;
    }

    public Boolean getIsCbow() {
        return isCbow;
    }

    public void setIsCbow(Boolean isCbow) {
        this.isCbow = isCbow;
    }

    public static void main(String[] args) throws IOException {
        Learn learn = new Learn();
        long start = System.currentTimeMillis() ;
        learn.learnFile(new File("InputFiles/xh.txt"));
        System.out.println("use time "+(System.currentTimeMillis()-start));
        learn.saveModel(new File("InputFiles/javaVector"));
        
    }
    class MyTask implements Runnable {
        private long nextRandom,nextRandom2,nextRandom3,nextRandom4;
        private List<WordNeuron> sentence = null,sentence2 = null,sentence3 = null,sentence4 = null;
        private short step = 0,step2 = 0,step3 = 0,step4 = 0;

        public void setArguments(Tri tri,Tri tri2,Tri tri3,Tri tri4){
            this.nextRandom = tri.nextRandom;
            this.sentence = tri.sentence;
            this.step = tri.step;
            this.nextRandom2 = tri2.nextRandom;
            this.sentence2 = tri2.sentence;
            this.step2 = tri.step;
            this.nextRandom3 = tri3.nextRandom;
            this.sentence3 = tri3.sentence;
            this.step3 = tri.step;
            this.nextRandom4 = tri4.nextRandom;
            this.sentence4 = tri4.sentence;
            this.step4 = tri.step;
        }
        @Override
        public void run() {
            while (true) {

                synchronized (taskList) {
                    // 如果仓库存储量不足
                    while (taskList.size() < 4) {
                        try {
                            taskList.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                    setArguments(taskList.poll(),taskList.poll(),taskList.poll(),taskList.poll());

                    taskList.notifyAll();
                }

                LearnByPiece( step, sentence, nextRandom);
                LearnByPiece( step2, sentence2, nextRandom2);
                LearnByPiece( step3, sentence3, nextRandom3);
                LearnByPiece( step4, sentence4, nextRandom4);
                sentence = null;
                sentence2 = null;
                sentence3 = null;
                sentence4 = null;
            }
        }
    }
    private void LearnByPiece(short step, List<WordNeuron> sentence, long nextRandom){
        for (int index = 0; index < sentence.size(); index++) {
            nextRandom = nextRandom * 25214903917L + 11;
            if (isCbow) {
                cbowGram(index, sentence, (int) nextRandom % window);
            } else {
                skipGram(index, sentence, (int) nextRandom % window);
            }
        }
    }
    class Tri {
        private long nextRandom;
        private List<WordNeuron> sentence = null;
        private short step;

        public Tri(long nextRandom, List<WordNeuron> sentence, short step) {
            this.nextRandom = nextRandom;
            this.sentence = sentence;
            this.step = step;
        }
    }

    class MyTaskThirdType implements Runnable {
        private long nextRandom,nextRandom2,nextRandom3,nextRandom4;
        private List<WordNeuron> sentence = null,sentence2 = null,sentence3 = null,sentence4 = null;
        private short step = 0,step2 = 0,step3 = 0,step4 = 0;

        public void setArguments(Tri tri,Tri tri2,Tri tri3,Tri tri4){
            this.nextRandom = tri.nextRandom;
            this.sentence = tri.sentence;
            this.step = tri.step;
            this.nextRandom2 = tri2.nextRandom;
            this.sentence2 = tri2.sentence;
            this.step2 = tri.step;
            this.nextRandom3 = tri3.nextRandom;
            this.sentence3 = tri3.sentence;
            this.step3 = tri.step;
            this.nextRandom4 = tri4.nextRandom;
            this.sentence4 = tri4.sentence;
            this.step4 = tri.step;
        }
        @Override
        public void run() {
            while (true) {

                synchronized (taskList) {
                    // 如果仓库存储量不足
                    while (taskList.size() < 4) {
                        try {
                            taskList.wait();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                    setArguments(taskList.poll(),taskList.poll(),taskList.poll(),taskList.poll());

                    taskList.notifyAll();
                }

                increLearnByPiece_Incrementally(step, sentence, nextRandom);
                increLearnByPiece_Incrementally(step2, sentence2, nextRandom2);
                increLearnByPiece_Incrementally(step3, sentence3, nextRandom3);
                increLearnByPiece_Incrementally(step4, sentence4, nextRandom4);
                sentence = null;
                sentence2 = null;
                sentence3 = null;
                sentence4 = null;
            }
        }
    }
    private void increLearnByPiece_Incrementally(short step, List<WordNeuron> sentence, long nextRandom){
        if(step == 0) {
            for (int index = 0; index < sentence.size(); index++) {
                nextRandom = nextRandom * 25214903917L + 11;
                if (isCbow) {
                    cbowGram_Incrementally(index, sentence, (int) nextRandom % window);
                } else {
                    skipGram_Incrementally(index, sentence, (int) nextRandom % window);
                }
            }
        }else{
            for (int index = 0; index < sentence.size(); index++) {
                nextRandom = nextRandom * 25214903917L + 11;
                if (isCbow) {
                    cbowGram(index, sentence, (int) nextRandom % window);
                } else {
                    skipGram(index, sentence, (int) nextRandom % window);
                }
            }
        }
    }
}
