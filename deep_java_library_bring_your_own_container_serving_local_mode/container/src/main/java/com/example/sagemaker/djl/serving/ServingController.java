package com.example.sagemaker.djl.serving;

import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.transform.CenterCrop;
import ai.djl.modality.cv.transform.Normalize;
import ai.djl.modality.cv.transform.Resize;
import ai.djl.modality.cv.transform.ToTensor;
import ai.djl.modality.cv.translator.ImageClassificationTranslator;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Translator;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.io.IOException;
import java.nio.file.Paths;


@RestController
public class ServingController {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private ZooModel model = null;

    Logger logger = LoggerFactory.getLogger(ServingController.class);

    @GetMapping(value = "/ping", produces = "application/json")
    public ResponseEntity<?> ping() {
        logger.info("ping - Start");

        try {
            if (model == null ) {
                logger.info("ping - initiating model");
                model = init_model();
            }

            logger.info("ping - OK");
            return new ResponseEntity<>("\n", HttpStatus.OK);
        } catch (Exception e) {
            logger.error("ping - Error: ",e);
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    @PostMapping(value = "/invocations", produces = "application/json")
    public ResponseEntity<?> invoke_model(@RequestBody String payload) {
        logger.info("invoke_model - Start - payload: {}",payload);

        try {
            String inputImageUrl = payload.replaceAll("\"", "");
            Image image = ImageFactory.getInstance().fromUrl(inputImageUrl);

            Predictor<Image, Classifications> predictor = model.newPredictor();
            Classifications classifications = predictor.predict(image);

            logger.info("invoke_model - Returning classifications: {}",classifications);
            return new ResponseEntity<>(classifications.toJson(), HttpStatus.OK);

        } catch (Exception e) {
            logger.error("invoke_model - Error: ",e);
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    private ZooModel init_model() throws ModelNotFoundException, MalformedModelException, IOException {
        logger.info("init_model - Start");

        Translator<Image, Classifications> translator = ImageClassificationTranslator.builder()
                .addTransform(new Resize(256))
                .addTransform(new CenterCrop(224, 224))
                .addTransform(new ToTensor())
                .addTransform(new Normalize(
                        new float[] {0.485f, 0.456f, 0.406f},
                        new float[] {0.229f, 0.224f, 0.225f}))
                .optApplySoftmax(true)
                .build();

        Criteria<Image, Classifications> criteria = Criteria.builder()
                .setTypes(Image.class, Classifications.class)
                .optModelPath(Paths.get("/opt/ml/model/build/pytorch_models/resnet18"))
                .optOption("mapLocation", "true") // this model requires mapLocation for GPU
                .optTranslator(translator)
                .optProgress(new ProgressBar()).build();

        ZooModel model = criteria.loadModel();

        logger.info("init_model - model loaded");
        return model;
    }
}
