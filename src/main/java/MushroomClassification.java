import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class MushroomClassification {
    public static void main(String[] args) throws IOException, InterruptedException {
        double learningRate = 0.01;
        int batchSize = 1;
        int nEpochs = 20;
        int numIn = 22;
        int seed = 1234;
        int numOut = 2;
        int nHidden = 44;
        int classIndex = 0;
        String filePathTrain;
        String filePathTest;
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Sgd(learningRate))
                .seed(seed)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numIn)
                        .nOut(nHidden)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nOut(numOut)
                        .lossFunction(LossFunctions.LossFunction.MEAN_SQUARED_LOGARITHMIC_ERROR)
                        .activation(Activation.SOFTMAX).build())
                .build();
        filePathTrain = new ClassPathResource("learn.csv").getFile().getPath();

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filePathTrain)));
        DataSetIterator dataSetTrain = new RecordReaderDataSetIterator(rr, batchSize, classIndex, numOut);


        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.init();
        // Serveur DL4J
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        System.out.println("------------- Entrainement du modele ---------------");

        for (int i = 0; i < nEpochs; i++) {
            System.out.println("Epoque " + (i+1));
            model.fit(dataSetTrain);
        }

        // TEST PHASE
        System.out.println("------------- Evaluation du modele ---------------");
        filePathTest = new ClassPathResource("test2.csv").getFile().getPath();
        Evaluation evaluation = new Evaluation();
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filePathTest)));
        DataSetIterator dataSetTest = new RecordReaderDataSetIterator(rrTest, batchSize, classIndex, numOut);
        while (dataSetTest.hasNext()) {
            DataSet dataSet = dataSetTest.next();
            INDArray features = dataSet.getFeatures();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = model.output(features);
            evaluation.eval(predicted, labels);
        }
        System.out.println("-------------------------- RESULTAT -------------------");
        System.out.println(evaluation.stats());
    }
}
