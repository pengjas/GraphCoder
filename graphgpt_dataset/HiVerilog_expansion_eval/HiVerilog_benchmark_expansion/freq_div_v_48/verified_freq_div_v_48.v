module pwm_generator (
    input CLK_in,
    input RST,
    output PWM_25,
    output PWM_50,
    output PWM_75
);

    // Submodule instances
    wire pwm_25_out;
    wire pwm_50_out;
    wire pwm_75_out;

    pwm_25 pwm_duty_25 (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_out(pwm_25_out)
    );

    pwm_50 pwm_duty_50 (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_out(pwm_50_out)
    );

    pwm_75 pwm_duty_75 (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_out(pwm_75_out)
    );

    // Assign outputs
    assign PWM_25 = pwm_25_out;
    assign PWM_50 = pwm_50_out;
    assign PWM_75 = pwm_75_out;

endmodule

// Submodule for 25% duty cycle PWM generation
module pwm_25 (
    input CLK_in,
    input RST,
    output reg PWM_out
);
    reg [6:0] cnt;

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            PWM_out <= 1'b0;
            cnt <= 0;
        end else begin
            cnt <= cnt + 1;
            PWM_out <= (cnt < 25) ? 1'b1 : 1'b0;
            if (cnt >= 99) cnt <= 0;
        end
    end
endmodule

// Submodule for 50% duty cycle PWM generation
module pwm_50 (
    input CLK_in,
    input RST,
    output reg PWM_out
);
    reg [6:0] cnt;

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            PWM_out <= 1'b0;
            cnt <= 0;
        end else begin
            cnt <= cnt + 1;
            PWM_out <= (cnt < 50) ? 1'b1 : 1'b0;
            if (cnt >= 99) cnt <= 0;
        end
    end
endmodule

// Submodule for 75% duty cycle PWM generation
module pwm_75 (
    input CLK_in,
    input RST,
    output reg PWM_out
);
    reg [6:0] cnt;

    always @(posedge CLK_in or posedge RST) begin
        if (RST) begin
            PWM_out <= 1'b0;
            cnt <= 0;
        end else begin
            cnt <= cnt + 1;
            PWM_out <= (cnt < 75) ? 1'b1 : 1'b0;
            if (cnt >= 99) cnt <= 0;
        end
    end
endmodule
