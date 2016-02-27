package main;

import io.DataSet;
import io.LabeledDataSet;
import io.MatProcessData;
import io.ParsedReviewData;

import java.io.*;
import java.util.Map;

import parallel.Parallel;
import parallel.ThreadPool;

import util.CommandLineUtils;
import classify.LabeledDatum;

public class Arguments {
	
	String dir = "data/parsed/"; 
	int minCount = DataSet.MINCOUNT;//
	boolean isTestLabelsKnown = false;//testlabel是否知道
	
	String ModelFile = null;//模型参数存储的文件
	String TreeDumpDir = null;//二叉树文件夹
	String WordMapFile = null;//词典文件
	String LabelMapFile = null;//标签文件
	String ClassifierFile = null;//分类器文件
    String featuresOutputFile = null;//特征输出文件
	String ProbabilitiesOutputFile = null;//
	
	//是否训练模型，
	boolean TrainModel = false, CurriculumLearning = false;
	//
    int NumFolds = 10, MaxIterations = 80, EmbeddingSize = 50, CatSize = 1;
    //字典的大小，隐藏层，显示层
	int DictionarySize, hiddenSize, visibleSize;
	double AlphaCat = 0.2, Beta = 0.5;
	double[] Lambda = new double[] { 1e-05, 0.0001, 1e-05, 0.01 };
	
	boolean exitOnReturn = false;

    //data和label
	LabeledDataSet<LabeledDatum<Integer, Integer>, Integer, Integer> Dataset = null;

	public void parseArguments(String[] args) throws IOException {
        System.out.println("Arguments 's parseArguments is running");
        Map<String, String> argMap = CommandLineUtils
				.simpleCommandLineParser(args);
		
		if (argMap.containsKey("-CurriculumLearning"))
		  CurriculumLearning = Boolean.parseBoolean(argMap.get("-CurriculumLearning"));

		if (argMap.containsKey("-minCount"))
			minCount = Integer.parseInt(argMap.get("-minCount"));
		
		if (argMap.containsKey("-NumFolds"))
			NumFolds = Integer.parseInt(argMap.get("-NumFolds")) - 1;

		if (argMap.containsKey("-MaxIterations"))
			MaxIterations = Integer.parseInt(argMap.get("-MaxIterations"));

		if (argMap.containsKey("-embeddingSize"))
			EmbeddingSize = Integer.parseInt(argMap.get("-embeddingSize"));

		if (argMap.containsKey("-alphaCat"))
			AlphaCat = Double.parseDouble(argMap.get("-alphaCat"));

		if (argMap.containsKey("-lambdaW"))
			Lambda[0] = Double.parseDouble(argMap.get("-lambdaW"));

		if (argMap.containsKey("-lambdaL"))
			Lambda[1] = Double.parseDouble(argMap.get("-lambdaL"));

		if (argMap.containsKey("-lambdaCat"))
			Lambda[2] = Double.parseDouble(argMap.get("-lambdaCat"));

		if (argMap.containsKey("-lambdaLRAE"))
			Lambda[3] = Double.parseDouble(argMap.get("-lambdaLRAE"));

		if (argMap.containsKey("-Beta"))
			Beta = Double.parseDouble(argMap.get("-Beta"));

		if (argMap.containsKey("-TrainModel"))
			TrainModel = Boolean.parseBoolean(argMap.get("-TrainModel"));

		if (argMap.containsKey("-WordMapFile"))
		{
			WordMapFile = argMap.get("-WordMapFile");
			if (!exists (WordMapFile))
			{	
				System.err.println ("Your WordMapFile points to an invalid file!");
				exitOnReturn = true;
				printUsage();
				return;
			}	
		}
		
		if (argMap.containsKey("-LabelMapFile"))
		{	
			LabelMapFile = argMap.get("-LabelMapFile");
			if (!exists (LabelMapFile))
			{	
				System.err.println ("Your LabelMapFile points to an invalid file!");
				exitOnReturn = true;
				printUsage();
				return;
			}	
		}
		
		if (argMap.containsKey("-ModelFile"))
			ModelFile = argMap.get("-ModelFile");
		else {
			System.err.println ("Please specify a ModelFile parameter.");
			exitOnReturn = true;
			printUsage();
			return;
		}
		
		if (argMap.containsKey("-ClassifierFile"))
			ClassifierFile = argMap.get("-ClassifierFile");
		else {
			System.err.println ("Please specify a ClassifierFile parameter.");
			exitOnReturn = true;
			printUsage();
			return;
		}
		
		if (argMap.containsKey("-FeaturesOutputFile"))
			featuresOutputFile = argMap.get("-FeaturesOutputFile");
		
		if (argMap.containsKey("-TreeDumpDir"))
		{
			TreeDumpDir = argMap.get("-TreeDumpDir");

			File treeDumpFile = new File (TreeDumpDir);
			if (!treeDumpFile.exists())
				treeDumpFile.mkdir();
			else if (!treeDumpFile.isDirectory())
			{
				System.err.println ("TreeDumpDir file exists but it is not a directory.");
				exitOnReturn = true;
				printUsage();
			}
		}
		
		if (argMap.containsKey("-ProbabilitiesOutputFile"))
			ProbabilitiesOutputFile = argMap.get("-ProbabilitiesOutputFile");
		
		if (!TrainModel && (ProbabilitiesOutputFile == null && featuresOutputFile == null)){
			System.err.println ("Please specify your output if you are not training.");
			exitOnReturn = true;
			printUsage();
			return;
		}

		if (argMap.containsKey("-NumCores"))
		{	
			int numCores = Integer.parseInt(argMap.get("-NumCores"));
			Parallel.setPoolSize(numCores);
			ThreadPool.setPoolSize(numCores);
		}

		if (argMap.containsKey("--help") || argMap.containsKey("-h")) {
			exitOnReturn = true;
			printUsage();
			return;
		}

		if (argMap.containsKey("-ProcessedDataDir")) {
			dir = argMap.get("-ProcessedDataDir");
			Dataset = new MatProcessData(dir);
		} else if (argMap.containsKey("-DataDir")) {
			dir = argMap.get("-DataDir");
			ParsedReviewData Data = new ParsedReviewData(dir,minCount,WordMapFile,LabelMapFile);
			if (Data.isTestLablesKnown())
			{
				isTestLabelsKnown = true;
				System.out.println("Test Lables known!");
			}
			Dataset = Data;
		} else
			Dataset = new MatProcessData(dir);

		CatSize = Dataset.getCatSize() - 1;
		DictionarySize = Dataset.Vocab.size();
		hiddenSize = EmbeddingSize;
		visibleSize = EmbeddingSize;

		System.out.println ("NumCategories : " + Dataset.getCatSize());
        System.out.println("Arguments 's parseArguments is finish");
    }

	public void printUsage() {
        System.out.println("Arguments 's printUsage is running");
        try {
			FileInputStream fstream = new FileInputStream("USAGE");
			DataInputStream in = new DataInputStream(fstream);
			BufferedReader br = new BufferedReader(new InputStreamReader(in));
			String strLine;

			while ((strLine = br.readLine()) != null) {
				System.out.println(strLine);
			}
			in.close();
		} catch (Exception e) {
			// Catch exception if any
			System.err.println("Error: " + e.getMessage());
		}
        System.out.println("Arguments 's printUsage is finish");
    }
	
	private boolean exists (String fileName)
	{
        System.out.println("Arguments 's exists is running");
        return new File (fileName).exists();
	}

}
