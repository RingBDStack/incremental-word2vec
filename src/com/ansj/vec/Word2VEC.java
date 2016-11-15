package com.ansj.vec;

import com.ansj.vec.domain.WordEntry;

import java.io.*;
import java.util.*;
import java.util.Map.Entry;

public class Word2VEC {
	private static boolean getVec = false;
	private static boolean global = false;
//	private static String globalTrainFilePath = "InputFiles/wiki.enLemmatize.4.text";
//	private static String globalModelFilePath = "InputFiles/javaSkip300Test7G";
//	private static String globalPlusFilePath = "InputFiles/javaSkip300Plus7G";
//	private static String globalTreeFilePath = "InputFiles/javaSkip300Tree7G";
//
//	private static String incrementalAddFilePath = "InputFiles/xh-added-5G.txt";
//	private static String incrementalModelFilePath = "InputFiles/javaSkip300Test2G+5G";
//	private static String incrementalPlusFilePath = "InputFiles/javaSkip300Plus2G+5G";
//	private static String incrementalTreeFilePath = "InputFiles/javaSkip300Tree2G+5G";
	public static void main(String[] args) throws IOException {
		if(args.length >= 1){
			global = Boolean.valueOf(args[0]);
		}
		long start = 0;
		Learn learn;
		String[] strs = {"台风", "地震", "战争", "谋杀", "霾", "疫情", "受贿", "习近平", "火灾", "紧急状态"};
		try {
			if(global) {
				start = System.currentTimeMillis();
				learn = new Learn();
				learn.learnFile(new File("InputFiles/wiki.enLemmatize.4.text"));
				learn.saveModel(new File("InputFiles/javaSkip300Test7G"));
				learn.saveModelPlus(new File("InputFiles/javaSkip300Plus7G"));
				learn.saveTreeNodes(new File("InputFiles/javaSkip300Tree7G"));
				System.out.println(  (System.currentTimeMillis() - start)  );

				testVec("InputFiles/javaSkip300Test7G",getVec,strs);
			}else {

				learn = new Learn();
				learn.learnFile_Incrementally(new File("InputFiles/wiki.enLemmatize.3.text"), new File("InputFiles/xh-added-5G.txt"),
						new File("InputFiles/javaSkip300Tree2G"), new File("InputFiles/javaSkip300Plus2G"));
				learn.saveModel(new File("InputFiles/javaSkip300Test2G+5G"));
				learn.saveModelPlus(new File("InputFiles/javaSkip300Plus2G+5G"));
				learn.saveTreeNodes(new File("InputFiles/javaSkip300Tree2G+5G"));

				testVec("InputFiles/javaSkip300Test2G+5G", getVec, strs);
			}
		}catch (Exception e) {
			e.printStackTrace();
		}
//		System.out.println(vec.analogy("男子", "女子", "男孩子"));
//		System.out.println("山西" + "\t" +
//		Arrays.toString(vec.getWordVector("山西")));
		// ;
		// System.out.println("毛泽东" + "\t" +
		// Arrays.toString(vec.getWordVector("毛泽东")));
		// ;
		// System.out.println("足球" + "\t" +
		// Arrays.toString(vec.getWordVector("足球")));

		// Word2VEC vec2 = new Word2VEC();
		// vec2.loadGoogleModel("InputFiles/vectors.bin") ;
		//
		//
//		String str = "拘留";
//		long start = System.currentTimeMillis();
//		//22779
//		for (int i = 0; i < 100; i++) {vec.distance(str);
//			//System.out.println(vec.distance(str));
//		}
//		System.out.println(System.currentTimeMillis() - start);

		// System.out.println(vec2.distance(str));
		//
		//
		// //男人 国王 女人
		// System.out.println(vec.analogy("邓小平", "毛泽东思想", "毛泽东"));
		// System.out.println(vec2.analogy("毛泽东", "毛泽东思想", "邓小平"));
	}

	private HashMap<String, float[]> wordMap = new HashMap<String, float[]>();

	private int words;
	private int size;
	private int topNSize = 30;

	/**
	 * 测试模型
	 *
	 * @param path
	 *            模型的路径
	 * @throws java.io.IOException
	 */
	public static void testVec(String path,boolean getVec,String[] strs) throws IOException {
		Word2VEC vec = new Word2VEC();
		vec.loadJavaModel(path);
		for (String str : strs) {
			if (getVec) {
				float[] tmp = vec.getWordVector(str);
				System.out.print(str + ":[");//distance(str));
				for (float each : tmp) {
					System.out.print(each + ",");
				}
				System.out.println("]");
			}
			else
				System.out.println(str + ":" + vec.distance(str));
		}
		//System.out.println("distance of \"男人\" & \"女人\" is " + vec.distanceOfWord("男人", "女人"));
		System.exit(-1);
	}

	/**
	 * 加载模型
	 * 
	 * @param path
	 *            模型的路径
	 * @throws java.io.IOException
	 */
	public void loadGoogleModel(String path) throws IOException {
		DataInputStream dis = null;
		BufferedInputStream bis = null;
		double len = 0;
		float vector = 0;
		try {
			bis = new BufferedInputStream(new FileInputStream(path));
			dis = new DataInputStream(bis);
			// //读取词数
			words = Integer.parseInt(readString(dis));
			// //大小
			size = Integer.parseInt(readString(dis));
			String word;
			float[] vectors = null;
			for (int i = 0; i < words; i++) {
				word = readString(dis);
				vectors = new float[size];
				len = 0;
				for (int j = 0; j < size; j++) {
					vector = readFloat(dis);
					len += vector * vector;
					vectors[j] = (float) vector;
				}
				len = Math.sqrt(len);

				for (int j = 0; j < size; j++) {
					vectors[j] /= len;
				}

				wordMap.put(word, vectors);
				dis.read();
			}
		} finally {
			bis.close();
			dis.close();
		}
	}

	/**
	 * 加载模型
	 * 
	 * @param path
	 *            模型的路径
	 * @throws java.io.IOException
	 */
	public void loadJavaModel(String path) throws IOException {
		try (DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(path)))) {
			words = dis.readInt();
			size = dis.readInt();

			float vector = 0;

			String key = null;
			float[] value = null;
			for (int i = 0; i < words; i++) {
				double len = 0;
				key = dis.readUTF();
				value = new float[size];
				for (int j = 0; j < size; j++) {
					vector = dis.readFloat();
					len += vector * vector;
					value[j] = vector;
				}

				len = Math.sqrt(len);
				//归一化
				for (int j = 0; j < size; j++) {
					value[j] /= len;
				}
				wordMap.put(key, value);
			}

		}
	}

	private static final int MAX_SIZE = 50;

	/**
	 * 近义词
	 * 
	 * @return
	 */
	public TreeSet<WordEntry> analogy(String word0, String word1, String word2) {
		float[] wv0 = getWordVector(word0);
		float[] wv1 = getWordVector(word1);
		float[] wv2 = getWordVector(word2);

		if (wv1 == null || wv2 == null || wv0 == null) {
			return null;
		}
		float[] wordVector = new float[size];
		for (int i = 0; i < size; i++) {
			wordVector[i] = wv1[i] - wv0[i] + wv2[i];
		}
		float[] tempVector;
		String name;
		List<WordEntry> wordEntrys = new ArrayList<WordEntry>(topNSize);
		for (Entry<String, float[]> entry : wordMap.entrySet()) {
			name = entry.getKey();
			if (name.equals(word0) || name.equals(word1) || name.equals(word2)) {
				continue;
			}
			float dist = 0;
			tempVector = entry.getValue();
			for (int i = 0; i < wordVector.length; i++) {
				dist += wordVector[i] * tempVector[i];
			}
			insertTopN(name, dist, wordEntrys);
		}
		return new TreeSet<WordEntry>(wordEntrys);
	}

	private void insertTopN(String name, float score, List<WordEntry> wordsEntrys) {
		// TODO Auto-generated method stub
		if (wordsEntrys.size() < topNSize) {
			wordsEntrys.add(new WordEntry(name, score));
			return;
		}
		float min = Float.MAX_VALUE;
		int minOffe = 0;
		for (int i = 0; i < topNSize; i++) {
			WordEntry wordEntry = wordsEntrys.get(i);
			if (min > wordEntry.score) {
				min = wordEntry.score;
				minOffe = i;
			}
		}

		if (score > min) {
			wordsEntrys.set(minOffe, new WordEntry(name, score));
		}

	}

	public float distanceOfWord(String wordA , String wordB ){
		float dist = 0;
		if(wordA.compareTo(wordB) == 0)
			return 1;
		float[] vectorA = getWordVector(wordA);
		float[] vectorB = getWordVector(wordB);
		if(vectorA == null || vectorB == null)
			return 0;
		for (int i = 0; i < vectorA.length; i++) {
			dist += vectorB[i] * vectorA[i];
		}
		return dist;
	}

	public int getMapsize(){
		return wordMap.size();
	}
	public Set<WordEntry> distance(String queryWord) {

		float[] center = wordMap.get(queryWord);
		if (center == null) {
			return Collections.emptySet();
		}

		int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		TreeSet<WordEntry> result = new TreeSet<WordEntry>();

		double min = Float.MIN_VALUE;
		for (Entry<String, float[]> entry : wordMap.entrySet()) {
			float[] vector = entry.getValue();
			float dist = 0;
			for (int i = 0; i < vector.length; i++) {
				dist += center[i] * vector[i];
			}

			if (dist > min) {
				result.add(new WordEntry(entry.getKey(), dist));
				if (resultSize < result.size()) {
					result.pollLast();
				}
				min = result.last().score;
			}
		}
		result.pollFirst();

		return result;
	}

	public Set<WordEntry> distance(List<String> words) {

		float[] center = null;
		for (String word : words) {
			center = sum(center, wordMap.get(word));
		}

		if (center == null) {
			return Collections.emptySet();
		}

		int resultSize = wordMap.size() < topNSize ? wordMap.size() : topNSize;
		TreeSet<WordEntry> result = new TreeSet<WordEntry>();

		double min = Float.MIN_VALUE;
		for (Entry<String, float[]> entry : wordMap.entrySet()) {
			float[] vector = entry.getValue();
			float dist = 0;
			for (int i = 0; i < vector.length; i++) {
				dist += center[i] * vector[i];
			}

			if (dist > min) {
				result.add(new WordEntry(entry.getKey(), dist));
				if (resultSize < result.size()) {
					result.pollLast();
				}
				min = result.last().score;
			}
		}
		result.pollFirst();

		return result;
	}

	private float[] sum(float[] center, float[] fs) {
		// TODO Auto-generated method stub

		if (center == null && fs == null) {
			return null;
		}

		if (fs == null) {
			return center;
		}

		if (center == null) {
			return fs;
		}

		for (int i = 0; i < fs.length; i++) {
			center[i] += fs[i];
		}

		return center;
	}

	/**
	 * 得到词向量
	 * 
	 * @param word
	 * @return
	 */
	public float[] getWordVector(String word) {
		return wordMap.get(word);
	}

	public static float readFloat(InputStream is) throws IOException {
		byte[] bytes = new byte[4];
		is.read(bytes);
		return getFloat(bytes);
	}

	/**
	 * 读取一个float
	 * 
	 * @param b
	 * @return
	 */
	public static float getFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}

	/**
	 * 读取一个字符串
	 * 
	 * @param dis
	 * @return
	 * @throws java.io.IOException
	 */
	private static String readString(DataInputStream dis) throws IOException {
		// TODO Auto-generated method stub
		byte[] bytes = new byte[MAX_SIZE];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) {
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[MAX_SIZE];
			}
		}
		sb.append(new String(bytes, 0, i + 1));
		return sb.toString();
	}

	public int getTopNSize() {
		return topNSize;
	}

	public void setTopNSize(int topNSize) {
		this.topNSize = topNSize;
	}

	public HashMap<String, float[]> getWordMap() {
		return wordMap;
	}

	public int getWords() {
		return words;
	}

	public int getSize() {
		return size;
	}

}
