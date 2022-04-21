package com.example.sagemaker.djl.serving;

import ai.djl.Application;
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooModel;
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


@RestController
public class ServingController {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private ZooModel<Image, DetectedObjects> model = null;

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
            Predictor<Image, DetectedObjects> predictor = model.newPredictor();
            Image image = ImageFactory.getInstance().fromUrl(inputImageUrl);
            DetectedObjects detection = predictor.predict(image);
            logger.info("detection: {}", detection);

            String json = GSON.toJson(detection);
            logger.info("invoke_model - Returning json: {}",json);
            return new ResponseEntity<>(json, HttpStatus.OK);

        } catch (Exception e) {
            logger.error("invoke_model - Error: ",e);
            return new ResponseEntity<>(null, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }


    private ZooModel<Image, DetectedObjects> init_model() throws ModelNotFoundException, MalformedModelException, IOException {
        logger.info("init_model - Start");
        String backbone;
        if ("TensorFlow".equals(Engine.getInstance().getEngineName())) {
            backbone = "mobilenet_v2";
        } else {
            backbone = "resnet50";
        }
        logger.info("backbone: " + backbone);

        Criteria<Image, DetectedObjects> criteria =
                Criteria.builder()
                        .optApplication(Application.CV.OBJECT_DETECTION)
                        .setTypes(Image.class, DetectedObjects.class)
                        .optFilter("backbone", backbone)
                        .build();

        ZooModel<Image, DetectedObjects> model = ModelZoo.loadModel(criteria);
        logger.info("init_model - model loaded");
        return model;
    }
}
