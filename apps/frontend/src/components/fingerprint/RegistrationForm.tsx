import { Input } from "@/components/ui/input";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

interface RegistrationFormProps {
  id: string;
  name: string;
  onIdChange: (val: string) => void;
  onNameChange: (val: string) => void;
}

export function RegistrationForm({
  id,
  name,
  onIdChange,
  onNameChange,
}: RegistrationFormProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Datos de Registro</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <Input
          placeholder="ID Personal"
          value={id}
          onChange={(e) => onIdChange(e.target.value)}
        />
        <Input
          placeholder="Nombre Completo"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
        />
      </CardContent>
    </Card>
  );
}
