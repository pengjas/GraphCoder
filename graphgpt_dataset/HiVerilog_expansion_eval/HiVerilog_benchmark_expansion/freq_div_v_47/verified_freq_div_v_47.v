module pwm_gen (
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

    pwm_25 pwm_module_25 (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_out(pwm_25_out)
    );

    pwm_50 pwm_module_50 (
        .CLK_in(CLK_in),
        .RST(RST),
        .PWM_out(pwm_50_out)
    );

    pwm_75 pwm_module_75 (
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
            PWM_out <= 0;
            cnt <= 0;
        end else begin
            if (cnt < 25)
                PWM_out <= 1;
            else
                PWM_out <= 0;

            cnt <= (cnt == 99) ? 0 : cnt + 1;
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
            PWM_out <= 0;
            cnt <= 0;
        end else begin
            if (cnt < 50)
                PWM_out <= 1;
            else
                PWM_out <= 0;

            cnt <= (cnt == 99) ? 0 : cnt + 1;
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
            PWM_out <= 0;
            cnt <= 0;
        end else begin
            if (cnt < 75)
                PWM_out <= 1;
            else
                PWM_out <= 0;

            cnt <= (cnt == 99) ? 0 : cnt + 1;
        end
    end
endmodule
