package com.github.exercise.util;

import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.s3.S3Client;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.ByteArrayInputStream;
import java.io.IOException;

public class S3Util {
    private String bucketName;
    private S3Client s3Client;

    public S3Util(final Region region, final String bucketName) {
        this.bucketName = bucketName;
        this.s3Client = S3Client.builder().region(region).build();
    }

    public void putObject(final String key, final byte[] bytes) throws IOException {
        final PutObjectRequest putObjectRequest = PutObjectRequest.builder().bucket(this.bucketName).key(key).build();
        this.s3Client.putObject(putObjectRequest,
                RequestBody.fromInputStream(new ByteArrayInputStream(bytes), Long.valueOf(bytes.length)));
    }


}
