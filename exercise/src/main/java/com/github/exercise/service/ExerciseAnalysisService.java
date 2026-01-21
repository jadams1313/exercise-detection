package com.github.exercise.service;

import com.github.exercise.data.Excercise;
import org.springframework.web.client.RestClient;

public class ExerciseAnalysisService implements AnalysisService{

    public ExerciseAnalysisService() {};

    public Excercise classifyExercise() {
        RestClient videoClassificationClient = new RestClient();

        //extend request to mL model api.
    }
}
