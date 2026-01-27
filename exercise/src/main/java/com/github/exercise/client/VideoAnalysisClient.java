package com.github.exercise.client;

import com.github.exercise.dto.VideoAnalysisRequest;
import com.github.exercise.dto.VideoAnalysisResponse;
import com.github.exercise.service.FileStorageService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.io.IOException;
import java.io.InputStream;
import java.util.Base64;
import static reactor.netty.http.HttpConnectionLiveness.log;

@Slf4j
@Service
@RequiredArgsConstructor
public class VideoAnalysisClient {
    private final RestTemplate restTemplate;
    private final FileStorageService fileStorageService;

    @Value("${video.analysis.api.url}")
    private String videoAnalysisApi;

    @Value("${video.analysis.api.timeout:300000}") // 5 minutes default
    private int timeout;

    public VideoAnalysisResponse analyzeVideo(String fileName) throws IOException {
        log.info("Fetching video from S3: {}", fileName);

        // Get video from S3
        InputStream videoStream = fileStorageService.getFileStream(fileName);
        byte[] videoBytes = videoStream.readAllBytes();
        String base64Video = Base64.getEncoder().encodeToString(videoBytes);

        VideoAnalysisRequest request = VideoAnalysisRequest();
        request.setVideoData(base64Video);
        request.setFileName(fileName);

        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        HttpEntity<VideoAnalysisRequest> entity = new HttpEntity<>(request, headers);

        try {
            log.info("Calling ML API for file: {}", fileName);
            ResponseEntity<VideoAnalysisResponse> response = restTemplate.exchange(
                    videoAnalysisApi + "/analyze",
                    HttpMethod.POST,
                    entity,
                    VideoAnalysisResponse.class
            );

            log.info("ML analysis completed for file: {}", fileName);
            return response.getBody();

        } catch (RestClientException e) {
            log.error("ML API call failed for file: {}", fileName, e);
            throw new RuntimeException("Failed to analyze video with ML model: " + e.getMessage(), e);
        }
    }
}
