module shift_reg (
    input clk,
    input d,
    output q
);

    wire u0, u1, u2;

    my_dff u0_inst (
        .clk(clk),
        .d(d),
        .q(u0)
    );

    my_dff u1_inst (
        .clk(clk),
        .d(u0),
        .q(u1)
    );

    my_dff u2_inst (
        .clk(clk),
        .d(u1),
        .q(u2)
    );

endmodule

module my_dff (
    input clk,
    input d,
    output q
);

    reg q_reg;

    always @(posedge clk) begin
        q_reg <= d;
    end

    assign q = q_reg;

endmodule