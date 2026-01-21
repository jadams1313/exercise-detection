package com.github.exercise.service;

import com.github.exercise.util.S3Util;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;

public class FileStorageService {
    private S3Util s3Util;
    private String bucketKey;
    public FileStorageService(final S3Util s3Util, final String bucketKey) {
        this.s3Util = s3Util;
        this.bucketKey = bucketKey;
    }

    public String storeFile(MultipartFile file) throws IOException {
        try {
            byte[] byteArrayOfVideo = file.getBytes();
            s3Util.putObject(bucketKey, byteArrayOfVideo);
        }
        catch(Exception e) {
            //start logging impl.
            throw new IllegalStateException(e);
        }
        return file.getOriginalFilename();
    }
}
