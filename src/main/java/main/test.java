package main;

import classify.LabeledDatum;
import classify.ReviewDatum;
import classify.ReviewFeatures;
import classify.StratifiedCrossValidation;
import io.LabeledDataSet;
import math.DifferentiableMatrixFunction;
import math.Norm1Tanh;
import parallel.Parallel;
import rae.FineTunableTheta;
import rae.LabeledRAETree;
import rae.RAECost;
import rae.RAEFeatureExtractor;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

/**
 * Created by Administrator on 2016/1/18.
 */
public class test {
    public static void main(String[] args) {
        System.out.println("test");
    }
}
