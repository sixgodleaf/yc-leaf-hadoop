package classify;

import io.DataSet;
import util.*;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * The class splits the data passed to it.
 * It is constructed with k and the data.
 */
public class CrossValidation<T extends Datum<F>,F> {
	protected int k, foldSize, totalSize;
	protected DataSet<T,F> Data;
	protected int[] permutation;
	
	/**
	 * Randomly permutes the data set and splits it into k folds
	 **/
	public CrossValidation(int k, DataSet<T,F> Data)
	{
		this.k = k;
		this.Data = Data;
		this.totalSize = Data.Data.size();
		this.foldSize = (int)Math.floor(this.totalSize / k);
		permutation = ArraysHelper.makeArray(0, this.totalSize-1);
		permutation = ArraysHelper.shuffle(permutation);
	}
	
	public CrossValidation(int k, DataSet<T,F> Data, int[] permutation)
	{
		this.k = k;
		this.Data = Data;
		this.totalSize = Data.Data.size();
		this.foldSize = (int)Math.floor(this.totalSize / k);
		this.permutation = permutation; 
	}

	protected void printSetBitmap(Collection<Integer> indices)
	{
        System.out.println("crassValidation 's printSetBitmap is running");
        int[] bitmap = new int[ totalSize ];
		for(int ti : indices )
			bitmap[ti] = 1;
				
		for(int bti : bitmap)
			System.out.printf("%d ",bti);
		System.out.println("crassValidation 's printSetBitmap is finish");
	}
	
	protected List<Integer> getTrainingIndices(int foldNumber)
	{
		return getTrainingIndices(foldNumber, k-1);
	}
	
	protected List<Integer> getTrainingIndices(int foldNumber, int numFolds)
	{
        System.out.println("crassValidation 's getTrainingIndices is running");
        List<Integer> trainingIndices = new ArrayList<Integer>( foldSize * numFolds );
		int i=0;
		while ( trainingIndices.size() < foldSize *numFolds )
		{
			if(i == foldNumber * foldSize )
			{
				i += foldSize;
				continue;
			}
			trainingIndices.add( permutation[i] );
			i++;
		}
		printSetBitmap(trainingIndices);
        System.out.println("crassValidation 's getTrainingIndices is finish");
        return trainingIndices;
	}

	protected List<Integer> getValidationIndices(int foldNumber)
	{
        System.out.println("crassValidation 's getValidationIndices is running");
        List<Integer> validationIndices = new ArrayList<Integer>( foldSize );
		int i = foldNumber * foldSize;
		while ( validationIndices.size() < foldSize )
		{
			validationIndices.add( permutation[i] );
			i++;
		}
		printSetBitmap(validationIndices);
        System.out.println("crassValidation 's getValidationIndices is finish");

        return validationIndices;
	}

	public List<T> getTrainingData(int foldNumber)
	{
		return getTrainingData(foldNumber, k-1);
	}
	
	public List<T> getTrainingData(int foldNumber, int numFolds)
	{
        System.out.println("crassValidation 's getTrainingData is running");
        List<Integer> trainingIndices = getTrainingIndices(foldNumber, numFolds);
		List<T> trainingData = new ArrayList<T>( trainingIndices.size() );
		for(int i : trainingIndices)
			trainingData.add( Data.Data.get(i) );
        System.out.println("crassValidation 's getTrainingData is finish");

        return trainingData;
	}

	public List<T> getValidationData(int foldNumber)
	{
        System.out.println("crassValidation 's getValidationData is running");
        List<Integer> validationIndices = getValidationIndices(foldNumber);
		List<T> validationData = new ArrayList<T>( validationIndices.size() );
		for(int i : validationIndices)
			validationData.add( Data.Data.get(i) );
        System.out.println("crassValidation 's getValidationData is finish");

        return validationData;
	}	
}
