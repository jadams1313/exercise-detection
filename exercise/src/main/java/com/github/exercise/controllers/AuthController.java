package com.github.exercise.controllers;

import com.github.exercise.dto.LoginRequest;
import com.github.exercise.dto.LoginResponse;
import com.github.exercise.dto.RegisterRequest;
import com.github.exercise.dto.UserResponse;
import com.github.exercise.data.User;
import com.github.exercise.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.validation.Valid;

@RestController
@RequestMapping("/api/auth")
@RequiredArgsConstructor
public class AuthController {

    private final UserService userService;

    @PostMapping("/register")
    public ResponseEntity<UserResponse> register(@Valid @RequestBody RegisterRequest request) {
        User user = userService.registerUser(
                request.getUsername(),
                request.getEmail(),
                request.getPassword()
        );

        UserResponse response = UserResponse.builder()
                .id(user.getId())
                .username(user.getUsername())
                .email(user.getEmail())
                .createdAt(user.getCreatedAt())
                .build();

        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @PostMapping("/login")
    public ResponseEntity<LoginResponse> login(@Valid @RequestBody LoginRequest request) {
        String token = userService.authenticateUser(request.getUsername(), request.getPassword());

        LoginResponse response = LoginResponse.builder()
                .token(token)
                .username(request.getUsername())
                .build();

        return ResponseEntity.ok(response);
    }
}