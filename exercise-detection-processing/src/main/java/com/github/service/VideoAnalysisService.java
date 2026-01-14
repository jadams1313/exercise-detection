package com.github.service;

import com.github.data.Excercise;

public class VideoAnalysisService implements AnalysisService{

    public VideoAnalysisService() {};

    public Excercise classifyExercise() {
        RestClient videoClassificationClient = new RestClient();

        //extend request to mL model api.
    }
}
